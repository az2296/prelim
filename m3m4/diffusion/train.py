from data import generate_samples
from model import make_denoiser, make_scheduler, make_denoising_batch, update_ema
from eval import eval_loss, eval_test_metrics
from parse import parse_args
import csv
from pathlib import Path
import sys
import torch
import torch.nn as nn
import copy
import math
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.hf_checkpoints import try_upload_checkpoint


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def make_optimizers(model, args):
    if args.optimizer == "adamw":
        return [torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )]

    muon_params, adamw_params = model.muon_param_groups()

    if args.optimizer == "normuon":
        from normuon import SingleDeviceNorMuon
        matrix_opt = SingleDeviceNorMuon(
            muon_params,
            lr=args.muon_lr,
            weight_decay=args.muon_weight_decay,
            momentum=args.muon_momentum,
            beta2=args.normuon_beta2,
        )
    else:
        matrix_opt = torch.optim.Muon(
            muon_params,
            lr=args.muon_lr,
            weight_decay=args.muon_weight_decay,
            momentum=args.muon_momentum,
            nesterov=args.muon_nesterov,
            ns_steps=args.muon_ns_steps,
            adjust_lr_fn=args.muon_adjust_lr_fn,
        )

    return [
        matrix_opt,
        torch.optim.AdamW(
            adamw_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        ),
    ]


def metrics_row(seed, test_metrics, total_time):
    row = {
        "seed": seed,
        "total_time": total_time,
    }

    for key, value in test_metrics.items():
        if key == "quantiles" or key.startswith("mse_quantile_"):
            continue

        row[key] = value

    for key, values in test_metrics.items():
        if not key.startswith("mse_quantile_"):
            continue
        target = key.removeprefix("mse_quantile_")
        for q, mse_q in zip(test_metrics["quantiles"], values):
            row[f"mse_quantile_{target}_{q:.2f}"] = mse_q

    return row


def main():
    
    args = parse_args()

    csv_file = open(args.csv, "w", newline="")
    writer = None
    device = torch.device(args.device)

    for seed in range(args.start_seed, args.end_seed+1):

        torch.manual_seed(seed)

        train, val, test = generate_samples(
            model=args.model,
            seed=seed,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
        )

        x_train, y_train = train
        x_val, y_val = val

        y_mean = y_train.mean(dim=0)
        y_std = y_train.std(dim=0)
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
        val = (x_val, y_val)

        model = make_denoiser(args.arch, t_dim=args.t_dim).to(device)
        ema_model = copy.deepcopy(model).to(device)
        scheduler = make_scheduler(args.num_train_timesteps)
        scheduler.config.clip_sample = False

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

        if args.ckpt is not None:
            torch.save(
                {
                    "model_state": ema_model.state_dict(),
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "arch": args.arch,
                    "t_dim": args.t_dim,
                    "num_train_timesteps": args.num_train_timesteps,
                    "num_inference_steps": args.num_inference_steps,
                    "data_model": args.model,
                    "seed": seed,
                },
                args.ckpt,
            )
            print(f"Seed {seed} | Saved checkpoint to {args.ckpt}")
            if args.hf_repo and not args.no_hf_upload:
                hf_prefix = args.hf_prefix or "m3m4/diffusion"
                seed_path = seed if args.start_seed != args.end_seed else None
                uploaded = try_upload_checkpoint(
                    args.ckpt,
                    repo_id=args.hf_repo,
                    repo_type=args.hf_repo_type,
                    private=args.hf_private,
                    prefix=hf_prefix,
                    seed=seed_path,
                )
                if uploaded is not None:
                    print(f"Seed {seed} | Uploaded checkpoint to {args.hf_repo}:{uploaded}")

        test_metrics = eval_test_metrics(
            model=ema_model,
            data=test,
            device=device,
            scheduler=scheduler,
            y_mean=y_mean,
            y_std=y_std,
            data_model=args.model,
            J=args.j_test,
            num_inference_steps=args.num_inference_steps,
        )

        row = metrics_row(seed, test_metrics, total_time)
        if writer is None:
            writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
            writer.writeheader()

        writer.writerow(row)
        csv_file.flush()

        if use_early_stopping:
            print(f"Seed {seed} | Best Val Loss: {best_val_loss:.4f}")
        else:
            print(f"Seed {seed} | Early stopping disabled; using final model")
        for target in ("y_1", "y_2"):
            print(f"Seed {seed} | Test MSE({target}): {test_metrics[f'mse_{target}']:.4f}")
            print(f"Seed {seed} | Test L1({target}): {test_metrics[f'l1_{target}']:.4f}")
            print(f"Seed {seed} | Test MSE(Mean, {target}): {test_metrics[f'mse_mean_{target}']:.4f}")
            print(f"Seed {seed} | Test MSE(sd, {target}): {test_metrics[f'mse_sd_{target}']:.4f}")
            for q, mse_q in zip(test_metrics["quantiles"], test_metrics[f"mse_quantile_{target}"]):
                print(f"Seed {seed} | Test MSE(q={q:.2f}, {target}): {mse_q:.4f}")
        print(f"Seed {seed} | Total training time: {total_time:.2f}s")

    csv_file.close()


if __name__ == "__main__":
    main()
