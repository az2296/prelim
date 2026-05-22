import copy
import os
from pathlib import Path
import sys
import time

import torch
from torchvision.utils import save_image

from data import get_dataloaders
from model import Critic, Generator, gradient_penalty, mc_generate
from parse import parse_args

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.hf_checkpoints import try_upload_checkpoint


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def make_image(x, y):
    image = x[:, :1].clone()
    image[:, :, 7:21, 7:21] = y
    return image


def center_mse(fake, real, mask):
    return ((fake - real).pow(2) * mask).sum() / mask.sum()


def save_samples(path, ema_generator, x, y, z_dim, device, n_z=3):
    masked = x[:, :1]
    mask = x[:, 1:2]
    rows = [masked]
    with torch.no_grad():
        for _ in range(n_z):
            z = torch.randn(x.shape[0], z_dim, device=device)
            rows.append(ema_generator(masked, mask, z))
    rows.append(make_image(x, y))
    grid = torch.stack(rows, dim=1).reshape(-1, 1, 28, 28)
    save_image(grid, path, nrow=len(rows), normalize=True, value_range=(-1, 1))


def make_optimizer(
    model,
    optimizer,
    lr,
    beta1,
    beta2,
    weight_decay,
    muon_lr,
    muon_weight_decay,
    muon_momentum,
    muon_nesterov,
    muon_ns_steps,
    muon_adjust_lr_fn,
):
    if optimizer == "adamw":
        return [
            torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=weight_decay,
            )
        ]

    muon_params, adamw_params = model.muon_param_groups()

    return [
        torch.optim.Muon(
            muon_params,
            lr=muon_lr,
            weight_decay=muon_weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            adjust_lr_fn=muon_adjust_lr_fn,
        ),
        torch.optim.AdamW(
            adamw_params,
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        ),
    ]


def save_checkpoint(path, generator, critic, ema_generator, g_optim, c_optim, args, epoch):
    torch.save(
        {
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "ema_generator": ema_generator.state_dict(),
            "g_optimizer": [opt.state_dict() for opt in g_optim],
            "c_optimizer": [opt.state_dict() for opt in c_optim],
            "args": vars(args),
            "epoch": epoch,
        },
        path,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader, _ = get_dataloaders(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    generator = Generator(args.z_dim, args.base_channels, args.leaky_slope)
    critic = Critic(args.base_channels, args.leaky_slope)
    generator = generator.to(device)
    critic = critic.to(device)
    ema_generator = copy.deepcopy(generator).eval()
    for param in ema_generator.parameters():
        param.requires_grad = False

    g_optim = make_optimizer(
        generator,
        args.optimizer_g,
        args.lr_g,
        args.beta1_g,
        args.beta2_g,
        args.weight_decay_g,
        args.muon_lr_g,
        args.muon_weight_decay_g,
        args.muon_momentum_g,
        args.muon_nesterov_g,
        args.muon_ns_steps_g,
        args.muon_adjust_lr_fn_g,
    )
    c_optim = make_optimizer(
        critic,
        args.optimizer_c,
        args.lr_c,
        args.beta1_c,
        args.beta2_c,
        args.weight_decay_c,
        args.muon_lr_c,
        args.muon_weight_decay_c,
        args.muon_momentum_c,
        args.muon_nesterov_c,
        args.muon_ns_steps_c,
        args.muon_adjust_lr_fn_c,
    )
    critic_step = 0

    os.makedirs(args.sample_dir, exist_ok=True)
    sample_x, sample_y = next(iter(val_loader))
    sample_x = sample_x[:8].to(device)
    sample_y = sample_y[:8].to(device)

    for epoch in range(1, args.epochs + 1):
        sync(device)
        start = time.time()

        generator.train()
        critic.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            masked = x[:, :1]
            mask = x[:, 1:2]
            real = make_image(x, y)
            batch_size = x.shape[0]

            for _ in range(args.n_critic):
                z = torch.randn(batch_size, args.z_dim, device=device)
                with torch.no_grad():
                    fake = generator(masked, mask, z)

                real_score = critic(masked, mask, real)
                fake_score = critic(masked, mask, fake)
                gp = torch.zeros((), device=device)
                critic_step += 1
                if critic_step % args.lazy_gp == 0:
                    gp = gradient_penalty(
                        critic,
                        masked,
                        mask,
                        real,
                        fake,
                        args.lambda_gp * args.lazy_gp,
                    )
                c_loss = fake_score.mean() - real_score.mean() + gp

                for opt in c_optim:
                    opt.zero_grad()
                c_loss.backward()
                for opt in c_optim:
                    opt.step()

            z = torch.randn(batch_size, args.z_dim, device=device)
            fake = generator(masked, mask, z)
            fake_score = critic(masked, mask, fake)
            adv_loss = -fake_score.mean()

            if args.recon_weight != 0:
                fake_rec = mc_generate(generator, masked, mask, args.z_dim, args.j_train)
                recon_loss = center_mse(fake_rec.mean(dim=0), real, mask)
                g_loss = (1 - args.recon_weight) * adv_loss + args.recon_weight * recon_loss
            else:
                g_loss = adv_loss

            for opt in g_optim:
                opt.zero_grad()
            g_loss.backward()
            for opt in g_optim:
                opt.step()
            update_ema(generator, ema_generator, args.ema_decay)

        val_mse = 0
        n_val = 0
        ema_generator.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                masked = x[:, :1]
                mask = x[:, 1:2]
                real = make_image(x, y)
                z = torch.randn(x.shape[0], args.z_dim, device=device)
                fake = ema_generator(masked, mask, z)
                val_mse += ((fake - real).pow(2) * mask).sum().item()
                n_val += mask.sum().item()

        sync(device)
        seconds = time.time() - start

        if epoch % args.print_every == 0:
            print(
                f"epoch {epoch}: "
                f"val_mse={val_mse / n_val:.6f} "
                f"time={seconds:.2f}s"
            )

        if epoch % args.sample_every == 0:
            save_samples(
                os.path.join(args.sample_dir, f"epoch_{epoch:04d}.png"),
                ema_generator,
                sample_x,
                sample_y,
                args.z_dim,
                device,
            )

    save_checkpoint(args.save_path, generator, critic, ema_generator, g_optim, c_optim, args, epoch)
    if args.hf_repo and not args.no_hf_upload:
        hf_prefix = args.hf_prefix or "mnist/wgan"
        uploaded = try_upload_checkpoint(
            args.save_path,
            repo_id=args.hf_repo,
            repo_type=args.hf_repo_type,
            private=args.hf_private,
            prefix=hf_prefix,
        )
        if uploaded is not None:
            print(f"Uploaded checkpoint to {args.hf_repo}:{uploaded}")


if __name__ == "__main__":
    main()
