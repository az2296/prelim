from data import generate_samples
from model import Generator, Critic, c_loss, g_loss, mc_generate, update_ema, gradient_penalty
from eval import eval_mse, eval_test_metrics, COORD_NAMES
from parse import parse_args
import csv
import torch
import copy
import math
import time


def sync_device(device):

    if device.type == "cuda":

        torch.cuda.synchronize()

    elif device.type == "mps":

        torch.mps.synchronize()


def make_optimizer(model, lr, beta1, beta2, weight_decay):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )


def metrics_row(seed, test_metrics, total_time):
    row = {"seed": seed, "total_time": total_time}
    for key, value in test_metrics.items():
        row[key] = value
    return row


def main():

    args = parse_args()

    csv_file = open(args.csv, "w", newline="")
    writer = None

    try:
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
        x_test, y_test = test

        x_mean = x_train.mean(dim=0)
        x_std = x_train.std(dim=0)
        x_std = torch.where(x_std > 0, x_std, torch.ones_like(x_std))
        x_train_std = (x_train - x_mean) / x_std

        y_mean = y_train.mean(dim=0)
        y_std = y_train.std(dim=0)
        y_train_std = (y_train - y_mean) / y_std

        train = (x_train_std, y_train_std)
        val = (x_val, y_val)
        test = (x_test, y_test)


        g = Generator(x_dim=x_train.shape[1], y_dim=y_train.shape[1], z_dim=args.z_dim, leaky_slope=args.leaky_slope)
        critic = Critic(x_dim=x_train.shape[1], y_dim=y_train.shape[1], leaky_slope=args.leaky_slope)


        g = g.to(device)
        critic = critic.to(device)

        g_ema = copy.deepcopy(g).to(device)


        g_optim = make_optimizer(g, args.lr_g, args.beta1_g, args.beta2_g, args.weight_decay_g)
        c_optim = make_optimizer(critic, args.lr_c, args.beta1_c, args.beta2_c, args.weight_decay_c)

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

                    with torch.no_grad():
                        fake_y = mc_generate(
                            g=g,
                            x=x_batch,
                            z_dim=args.z_dim,
                            J=1
                        )

                    fake_y_flat = fake_y.reshape(-1, y_batch.shape[1]).detach()

                    fake_scores = critic(x_batch, fake_y_flat)
                    real_scores = critic(x_batch, y_batch)

                    loss_c = c_loss(real_scores, fake_scores) + gradient_penalty(
                        critic=critic,
                        x_real=x_batch,
                        y_real=y_batch,
                        x_fake=x_batch,
                        y_fake=fake_y_flat,
                        lambda_gp=args.lambda_gp
                    )

                    loss_c.backward()
                    c_optim.step()


                g_optim.zero_grad(set_to_none=True)

                fake_y_adv = mc_generate(
                    g=g,
                    x=x_batch,
                    z_dim=args.z_dim,
                    J=1
                )
                fake_scores = critic(x_batch, fake_y_adv.reshape(-1, y_batch.shape[1]))

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
                g_optim.step()

                update_ema(g, g_ema, decay=args.ema_decay)

            sync_device(device)
            epoch_time = time.time() - epoch_start

            if use_early_stopping:
                val_mse = eval_mse(
                    g=g_ema,
                    data=val,
                    device=device,
                    x_mean=x_mean,
                    x_std=x_std,
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

        eval_device = torch.device(args.eval_device) if args.eval_device is not None else device
        g_ema = g_ema.to(eval_device)

        test_metrics = eval_test_metrics(
            g=g_ema,
            data=test,
            device=eval_device,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            z_dim=args.z_dim,
            J=args.j_test
        )

        row = metrics_row(seed, test_metrics, total_time)
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
        csv_file.flush()

        if use_early_stopping:
            print(f"Seed {seed} | Best Val MSE: {best_val_mse:.4f}")
        else:
            print(f"Seed {seed} | Early stopping disabled; using final EMA model")
        for name in COORD_NAMES:
            print(f"Seed {seed} | Test MSE({name}): {test_metrics[f'mse_{name}']:.4f}")
            print(f"Seed {seed} | Test L1({name}): {test_metrics[f'l1_{name}']:.4f}")
            print(f"Seed {seed} | Test CP({name}): {test_metrics[f'cp_{name}']:.4f}")
            print(f"Seed {seed} | Test LPI({name}): {test_metrics[f'lpi_{name}']:.4f}")
        print(f"Seed {seed} | Total training time: {total_time:.2f}s")
    finally:
        csv_file.close()

if __name__ == "__main__":
    main()
