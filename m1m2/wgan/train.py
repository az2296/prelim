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


def make_optimizers(
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
        return [torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )]

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



            g_optims = make_optimizers(
                g,
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
            c_optims = make_optimizers(
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

            g_schedulers = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=args.lr_decay_g) for o in g_optims]
            c_schedulers = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=args.lr_decay_c) for o in c_optims]


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
                        for o in c_optims:
                            o.zero_grad(set_to_none=True)

                        with torch.no_grad():
                            fake_y = mc_generate(
                                g=g,
                                x=x_batch,
                                z_dim=args.z_dim,
                                J=1
                            )

                        fake_y_flat = fake_y.reshape(-1, 1).detach()

                        fake_scores = critic(x_batch, fake_y_flat)
                        real_scores = critic(x_batch, y_batch)

                        loss_c = c_loss(real_scores, fake_scores)+ gradient_penalty(
                            critic=critic,
                            x_real=x_batch,
                            y_real=y_batch,
                            x_fake=x_batch,
                            y_fake=fake_y_flat,
                            lambda_gp=args.lambda_gp
                        )

                        loss_c.backward()
                        for o in c_optims:
                            o.step()


                    for o in g_optims:
                        o.zero_grad(set_to_none=True)

                    fake_y_adv = mc_generate(
                        g=g,
                        x=x_batch,
                        z_dim=args.z_dim,
                        J=1
                    )
                    fake_scores = critic(x_batch, fake_y_adv.reshape(-1, 1))

                    fake_y_rec = None
                    if args.recon_weight != 0:
                        fake_y_rec = mc_generate(
                            g=g,
                            x=x_batch,
                            z_dim=args.z_dim,
                            J=args.j_train
                        )

                    loss_g = g_loss(
                        fake_scores=fake_scores,
                        fake_y=fake_y_rec,
                        y=y_batch,
                        recon_weight=args.recon_weight
                    )

                    loss_g.backward()
                    for o in g_optims:
                        o.step()

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

                for s in g_schedulers:
                    s.step()
                for s in c_schedulers:
                    s.step()
            
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
