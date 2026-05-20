from data import generate_samples
from model import Denoiser, make_scheduler, make_denoising_batch
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


def make_optimizers(model, args):
    if args.optimizer == "adamw":
        return [torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )]

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
        if key in {"quantiles", "mse_quantile"}:
            continue

        row[key] = value

    for q, mse_q in zip(test_metrics["quantiles"], test_metrics["mse_quantile"]):
        row[f"mse_quantile_{q:.2f}"] = mse_q

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
            d=args.x_dim,
            seed=seed,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
        )

        x_train, y_train = train
        x_val, y_val = val

        y_mean = y_train.mean()
        y_std = y_train.std()
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
        val = (x_val, y_val)

        model = Denoiser(x_dim=args.x_dim, t_dim=args.t_dim).to(device)
        scheduler = make_scheduler(args.num_train_timesteps)
        scheduler.config.clip_sample = False

        optims = make_optimizers(model, args)

        best_val_loss = math.inf
        bad_epochs = 0
        best_model_state = copy.deepcopy(model.state_dict())
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

            sync_device(device)
            epoch_time = time.time() - epoch_start

            if use_early_stopping:
                val_loss = eval_loss(
                    model=model,
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
                    best_model_state = copy.deepcopy(model.state_dict())

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
            model.load_state_dict(best_model_state)

        test_metrics = eval_test_metrics(
            model=model,
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
        print(f"Seed {seed} | Test MSE: {test_metrics['mse']:.4f}")
        print(f"Seed {seed} | Test L1: {test_metrics['l1']:.4f}")
        print(f"Seed {seed} | Test MSE(Mean): {test_metrics['mse_mean']:.4f}")
        print(f"Seed {seed} | Test MSE(sd): {test_metrics['mse_sd']:.4f}")
        for q, mse_q in zip(test_metrics['quantiles'], test_metrics['mse_quantile']):
            print(f"Seed {seed} | Test MSE(q={q:.2f}): {mse_q:.4f}")
        print(f"Seed {seed} | Total training time: {total_time:.2f}s")

    csv_file.close()


if __name__ == "__main__":
    main()
