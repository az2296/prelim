from data import generate_samples
from model import Generator, Critic, c_loss, g_loss, mc_generate, update_ema, gradient_penalty
from eval import eval_mse, eval_test_metrics
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

    try:
        for seed in range(args.start_seed, args.end_seed+1):

            device = torch.device(args.device)

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
            x_test, y_test = test

            y_mean = y_train.mean()
            y_std = y_train.std()
            y_train_std = (y_train - y_mean) / y_std

            train = (x_train, y_train_std)
            val = (x_val, y_val)
            test = (x_test, y_test)


            g = Generator(x_dim = args.x_dim, z_dim = args.z_dim, leaky_slope= args.leaky_slope)
            critic = Critic(x_dim=args.x_dim, leaky_slope= args.leaky_slope)


            g = g.to(device)
            critic = critic.to(device)

            g_ema = copy.deepcopy(g).to(device)



            g_optim = torch.optim.AdamW(g.parameters(), lr = args.lr_g, betas = (args.beta1_g, args.beta2_g), weight_decay = args.weight_decay_g)
            c_optim = torch.optim.AdamW(critic.parameters(), lr = args.lr_c, betas = (args.beta1_c, args.beta2_c), weight_decay = args.weight_decay_c)

            g_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_optim, gamma=args.lr_decay_g)
            c_scheduler = torch.optim.lr_scheduler.ExponentialLR(c_optim, gamma=args.lr_decay_c)


            best_val_mse = math.inf
            bad_epochs = 0
            use_early_stopping = args.patience < args.n_epochs

            loader_rng = torch.Generator()
            loader_rng.manual_seed(seed)

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train[0], train[1]),
                batch_size=args.batch_size,
                shuffle=True,
                generator = loader_rng
            )

            g = g.to(device)
            critic = critic.to(device)

            g_ema = copy.deepcopy(g).to(device)

            sync_device(device)
            total_start = time.time()

            for epoch in range(args.n_epochs):
                sync_device(device)
                epoch_start = time.time()

                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)


                    for _ in range(args.n_critic):
                        c_optim.zero_grad(set_to_none=True)

                        fake_y = mc_generate(
                            g=g,
                            x=x_batch,
                            z_dim=args.z_dim,
                            J=args.j_train
                        )  

                        x_rep = x_batch.repeat(args.j_train, 1)
                        y_rep = y_batch.repeat(args.j_train, 1)
                        fake_y_flat = fake_y.reshape(-1, 1).detach()

                        fake_scores = critic(x_rep, fake_y_flat)
                        real_scores = critic(x_batch, y_batch)

                        loss_c = c_loss(real_scores, fake_scores)+ gradient_penalty(
                            critic=critic,
                            x_real=x_rep,
                            y_real=y_rep,
                            x_fake=x_rep,
                            y_fake=fake_y_flat,
                            lambda_gp=args.lambda_gp
                        )

                        loss_c.backward()
                        c_optim.step()


                    g_optim.zero_grad(set_to_none=True)

                    fake_y = mc_generate(
                        g=g,
                        x=x_batch,
                        z_dim=args.z_dim,
                        J=args.j_train
                    )  

                    x_rep = x_batch.repeat(args.j_train, 1)

                    fake_scores = critic(x_rep, fake_y.reshape(-1, 1))

                    loss_g = g_loss(
                        fake_scores=fake_scores,
                        fake_y=fake_y,
                        y=y_batch,
                        recon_weight=args.recon_weight
                    )

                    loss_g.backward()
                    g_optim.step()

                    update_ema(g, g_ema, decay=args.ema_decay)

                
                sync_device(device)
                epoch_time = time.time() - epoch_start

                if use_early_stopping:
                    val_mse = eval_mse(
                    g=g_ema,
                    data=val,
                    device=device,
                    y_mean=y_mean,
                    y_std=y_std,
                    z_dim=args.z_dim,
                    J=args.j_eval
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

                        best_g_ema_state = copy.deepcopy(g_ema.state_dict())

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

                g_scheduler.step()
                c_scheduler.step()
            
            sync_device(device)
            total_time = time.time() - total_start


            if use_early_stopping:
                g_ema.load_state_dict(best_g_ema_state)


            test_metrics = eval_test_metrics(
                g=g_ema,
                data=test,
                device=device,
                y_mean=y_mean,
                y_std=y_std,
                model=args.model,
                z_dim=args.z_dim,
                J=args.j_test
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
            print(f"Seed {seed} | Test MSE(sd): {test_metrics['mse_sd']:.4f}")
            for q, mse_q in zip(test_metrics['quantiles'], test_metrics['mse_quantile']):
                print(f"Seed {seed} | Test MSE(q={q:.2f}): {mse_q:.4f}")
            print(f"Seed {seed} | Total training time: {total_time:.2f}s")
    finally:
        csv_file.close()

if __name__ == "__main__":
    main()
