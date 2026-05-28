from data import generate_samples
from model import make_denoiser, make_scheduler, make_denoising_batch, update_ema
from eval import eval_loss, eval_test_metrics
from parse import parse_args
import csv
import torch
import torch.nn as nn
import copy
import math
import time


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def split_decay(params):
    decay, no_decay = [], []
    for p in params:
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return decay, no_decay


def make_adamw(params, args):
    decay, no_decay = split_decay(params)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )


def make_optimizers(model, args):
    if args.optimizer == "adamw":
        return [make_adamw(model.parameters(), args)]

    muon_params, adamw_params = model.muon_param_groups()

    return [
        torch.optim.Muon(
            muon_params,
            lr=args.muon_lr,
            weight_decay=args.muon_weight_decay,
            momentum=args.muon_momentum,
            nesterov=args.muon_nesterov,
            ns_steps=args.muon_ns_steps,
            adjust_lr_fn=args.muon_adjust_lr_fn,
        ),
        make_adamw(adamw_params, args),
    ]


def metrics_row(seed, test_metrics, total_time):
    row = {"seed": seed, "total_time": total_time}
    for key, value in test_metrics.items():
        row[key] = value
    return row


def main():

    args = parse_args()

    csv_file = open(args.csv, "w", newline="")
    writer = None
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)

    train, val, test = generate_samples(
        seed=seed,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
    )

    x_train, y_train = train
    x_val, y_val = val

    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0)
    x_std = torch.where(x_std > 0, x_std, torch.ones_like(x_std))
    x_train = (x_train - x_mean) / x_std

    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0)
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    val = (x_val, y_val)

    model = make_denoiser(args.arch, t_dim=args.t_dim).to(device)
    ema_model = copy.deepcopy(model).to(device)
    scheduler = make_scheduler(args.num_train_timesteps)
    scheduler.config.clip_sample = True
    scheduler.config.clip_sample_range = 3.0

    optims = make_optimizers(model, args)

    best_val_loss = math.inf
    bad_epochs = 0
    best_model_state = copy.deepcopy(ema_model.state_dict())
    use_early_stopping = args.patience < args.n_epochs

    loader_rng = torch.Generator()
    loader_rng.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_rng,
    )

    sync_device(device)
    total_start = time.time()

    for epoch in range(args.n_epochs):
        sync_device(device)
        epoch_start = time.time()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            x_batch, y_t, t, noise = make_denoising_batch(x_batch, y_batch, scheduler)
            noise_pred = model(x_batch, y_t, t)
            loss = nn.functional.mse_loss(noise_pred, noise)

            for optim in optims:
                optim.zero_grad(set_to_none=True)
            loss.backward()
            for optim in optims:
                optim.step()

            update_ema(model, ema_model, decay=args.ema_decay)

        sync_device(device)
        epoch_time = time.time() - epoch_start

        if use_early_stopping:
            val_loss = eval_loss(
                model=ema_model,
                data=val,
                device=device,
                scheduler=scheduler,
                x_mean=x_mean,
                x_std=x_std,
                J=args.j_val,
            )

            if (epoch + 1) % args.print_every == 0 or epoch == 0:
                print(
                    f"Seed {seed} | "
                    f"Epoch {epoch + 1:03d} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Train time: {epoch_time:.2f}s"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                best_model_state = copy.deepcopy(ema_model.state_dict())

            else:
                bad_epochs += 1

            if bad_epochs >= args.patience:
                print(
                    f"Seed {seed} | "
                    f"Early stopping at epoch {epoch + 1} | "
                    f"Best Val Loss: {best_val_loss:.4f}"
                )
                break

        else:
            if (epoch + 1) % args.print_every == 0 or epoch == 0:
                print(
                    f"Seed {seed} | "
                    f"Epoch {epoch + 1:03d} | "
                    f"Train time: {epoch_time:.2f}s"
                )

    sync_device(device)
    total_time = time.time() - total_start

    if use_early_stopping:
        ema_model.load_state_dict(best_model_state)

    eval_device = torch.device(args.eval_device) if args.eval_device is not None else device
    ema_model = ema_model.to(eval_device)

    test_metrics = eval_test_metrics(
        model=ema_model,
        data=test,
        device=eval_device,
        scheduler=scheduler,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        J=args.j_test,
        num_inference_steps=args.num_inference_steps,
    )

    row = metrics_row(seed, test_metrics, total_time)
    writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)
    csv_file.flush()

    if use_early_stopping:
        print(f"Seed {seed} | Best Val Loss: {best_val_loss:.4f}")
    else:
        print(f"Seed {seed} | Early stopping disabled; using final model")
    print(f"Seed {seed} | Test MSE: {test_metrics['mse']:.4f}")
    print(f"Seed {seed} | Test L1: {test_metrics['l1']:.4f}")
    print(f"Seed {seed} | Test LPI: {test_metrics['lpi']:.4f}")
    print(f"Seed {seed} | Test CP: {test_metrics['cp']:.4f}")
    print(f"Seed {seed} | Test SD-UBE: {test_metrics['sd_ube']:.4f}")
    print(f"Seed {seed} | Test SD-LBE: {test_metrics['sd_lbe']:.4f}")
    print(f"Seed {seed} | Total training time: {total_time:.2f}s")

    csv_file.close()


if __name__ == "__main__":
    main()
