import copy
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from data import get_dataloaders
from model import Denoiser, make_noisy_batch, make_scheduler
from parse import parse_args

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.hf_checkpoints import try_upload_checkpoint


def loss_fn(pred, noise):
    return F.mse_loss(pred[:, :, 7:21, 7:21], noise[:, :, 7:21, 7:21])


def update_ema(model, ema_model, decay):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()


def make_optimizer(model, args):
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        return [optimizer]

    muon_params, adamw_params = model.muon_param_groups()

    muon = torch.optim.Muon(
        muon_params,
        lr=args.muon_lr,
        weight_decay=args.muon_weight_decay,
        momentum=args.muon_momentum,
        nesterov=args.muon_nesterov,
        ns_steps=args.muon_ns_steps,
        adjust_lr_fn=args.muon_adjust_lr_fn,
    )
    adamw = torch.optim.AdamW(
        adamw_params,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    return [muon, adamw]


def make_ema(model, args):
    if args.no_ema:
        return None

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False
    return ema_model


def save_checkpoint(path, model, ema_model, optimizer, args, epoch, best_val):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": [opt.state_dict() for opt in optimizer],
        "epoch": epoch,
        "best_val": best_val,
        "args": vars(args),
    }

    if ema_model is not None:
        checkpoint["ema_model"] = ema_model.state_dict()

    torch.save(checkpoint, path)


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

    scheduler = make_scheduler(args.train_steps)
    model = Denoiser(t_dim=args.t_dim).to(device)
    optimizer = make_optimizer(model, args)
    ema_model = make_ema(model, args)

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        sync(device)
        start = time.time()

        model.train()
        train_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            denoiser_input, noise, t = make_noisy_batch(x, y, scheduler)
            pred = model(denoiser_input, t)
            loss = loss_fn(pred, noise)

            for opt in optimizer:
                opt.zero_grad()
            loss.backward()
            for opt in optimizer:
                opt.step()

            if ema_model is not None:
                update_ema(model, ema_model, args.ema_decay)

            train_loss += loss.item() * x.shape[0]

        train_loss = train_loss / len(train_loader.dataset)

        val_model = ema_model if ema_model is not None else model
        val_model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                denoiser_input, noise, t = make_noisy_batch(x, y, scheduler)
                pred = val_model(denoiser_input, t)
                loss = loss_fn(pred, noise)

                val_loss += loss.item() * x.shape[0]

        val_loss = val_loss / len(val_loader.dataset)

        sync(device)
        seconds = time.time() - start

        if epoch % args.print_every == 0:
            print(
                f"epoch {epoch}: "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"time={seconds:.2f}s"
            )

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                args.save_path, model, ema_model, optimizer, args, epoch, best_val
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"early stopping after {epoch} epochs")
            break

    if args.hf_repo and not args.no_hf_upload:
        hf_prefix = args.hf_prefix or "mnist/diffusion"
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
