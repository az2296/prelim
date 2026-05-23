from data import generate_samples
from eval import eval_mse, eval_test_metrics
from model import Regressor, update_ema
from parse import parse_args
import copy
import csv
import math
import time
import torch
import torch.nn as nn


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def adamw_param_groups(params, weight_decay):
    decay, no_decay = [], []
    for p in params:
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def make_optimizers(model, args):
    if args.optimizer == "adamw":
        return [torch.optim.AdamW(
            adamw_param_groups(model.parameters(), args.weight_decay),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
        )]

    muon_params, adamw_params = model.muon_param_groups()

    optims = []
    if muon_params:
        optims.append(
            torch.optim.Muon(
                muon_params,
                lr=args.muon_lr,
                weight_decay=args.muon_weight_decay,
                momentum=args.muon_momentum,
                nesterov=args.muon_nesterov,
                ns_steps=args.muon_ns_steps,
                adjust_lr_fn=args.muon_adjust_lr_fn,
            )
        )
    if adamw_params:
        optims.append(
            torch.optim.AdamW(
                adamw_param_groups(adamw_params, args.weight_decay),
                lr=args.lr,
                betas=(args.beta1, args.beta2),
            )
        )
    return optims


def metrics_row(seed, test_metrics, total_time):
    row = {
        "seed": seed,
        "total_time": total_time,
    }

    for key, value in test_metrics.items():
        row[key] = value

    return row


def main():

    args = parse_args()

    csv_file = open(args.csv, "w", newline="")
    writer = None
    device = torch.device(args.device)

    try:
        for seed in range(args.start_seed, args.end_seed + 1):

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

            y_mean = y_train.mean()
            y_std = y_train.std()
            y_train_std = (y_train - y_mean) / y_std
            train = (x_train, y_train_std)

            model = Regressor(x_dim=args.x_dim).to(device)
            ema_model = copy.deepcopy(model).to(device)

            optims = make_optimizers(model, args)
            schedulers = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=args.lr_decay) for o in optims]

            best_val_mse = math.inf
            bad_epochs = 0
            best_model_state = copy.deepcopy(ema_model.state_dict())
            use_early_stopping = args.patience < args.n_epochs

            loader_rng = torch.Generator()
            loader_rng.manual_seed(seed)

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train[0], train[1]),
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

                    pred_y = model(x_batch)
                    loss = nn.functional.mse_loss(pred_y, y_batch)

                    for optim in optims:
                        optim.zero_grad(set_to_none=True)
                    loss.backward()
                    for optim in optims:
                        optim.step()

                    update_ema(model, ema_model, decay=args.ema_decay)

                sync_device(device)
                epoch_time = time.time() - epoch_start

                if use_early_stopping:
                    val_mse = eval_mse(
                        model=ema_model,
                        data=val,
                        device=device,
                        y_mean=y_mean,
                        y_std=y_std,
                    )

                    if (epoch + 1) % args.print_every == 0 or epoch == 0:
                        print(
                            f"Seed {seed} | "
                            f"Epoch {epoch + 1:03d} | "
                            f"Val MSE: {val_mse:.4f} | "
                            f"Train time: {epoch_time:.2f}s"
                        )

                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        bad_epochs = 0
                        best_model_state = copy.deepcopy(ema_model.state_dict())

                    else:
                        bad_epochs += 1

                    if bad_epochs >= args.patience:
                        print(
                            f"Seed {seed} | "
                            f"Early stopping at epoch {epoch + 1} | "
                            f"Best Val MSE: {best_val_mse:.4f}"
                        )
                        break

                else:
                    if (epoch + 1) % args.print_every == 0 or epoch == 0:
                        print(
                            f"Seed {seed} | "
                            f"Epoch {epoch + 1:03d} | "
                            f"Train time: {epoch_time:.2f}s"
                        )

                for scheduler in schedulers:
                    scheduler.step()

            sync_device(device)
            total_time = time.time() - total_start

            if use_early_stopping:
                ema_model.load_state_dict(best_model_state)

            test_metrics = eval_test_metrics(
                model=ema_model,
                data=test,
                device=device,
                y_mean=y_mean,
                y_std=y_std,
                data_model=args.model,
            )

            row = metrics_row(seed, test_metrics, total_time)
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                writer.writeheader()

            writer.writerow(row)
            csv_file.flush()

            if use_early_stopping:
                print(f"Seed {seed} | Best Val MSE: {best_val_mse:.4f}")
            else:
                print(f"Seed {seed} | Early stopping disabled; using final EMA model")
            print(f"Seed {seed} | Test MSE: {test_metrics['mse']:.4f}")
            print(f"Seed {seed} | Test L1: {test_metrics['l1']:.4f}")
            print(f"Seed {seed} | Test MSE(Mean): {test_metrics['mse_mean']:.4f}")
            print(f"Seed {seed} | Total training time: {total_time:.2f}s")
    finally:
        csv_file.close()


if __name__ == "__main__":
    main()
