from model import mc_generate

import torch
import torch.nn as nn


@torch.no_grad()
def eval_mse(g, data, device, x_mean, x_std, y_mean, y_std, z_dim=100, J=20):
    x, y_true = data
    x = x.to(device)
    y_true = y_true.to(device).squeeze(1)

    x_std_input = (x - x_mean.to(device)) / x_std.to(device)
    fake_y_std = mc_generate(g, x_std_input, z_dim=z_dim, J=J)
    mean_fake_y_std = fake_y_std.mean(dim=0)

    pred_y = mean_fake_y_std * y_std.to(device) + y_mean.to(device)

    mse = nn.functional.mse_loss(pred_y, y_true)
    return mse.item()


@torch.no_grad()
def eval_test_metrics(
    g,
    data,
    device,
    x_mean,
    x_std,
    y_mean,
    y_std,
    z_dim=100,
    J=1000,
):
    x, y_true = data
    x = x.to(device)
    y_true = y_true.to(device).squeeze(1)
    x_mean = x_mean.to(device)
    x_std = x_std.to(device)
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)
    quantiles = torch.tensor((0.025, 0.975), device=device, dtype=x.dtype)

    x_std_input = (x - x_mean) / x_std

    chunk_size = 500
    n = x_std_input.shape[0]
    fake_y = torch.empty(J, n, device=device, dtype=x.dtype)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = mc_generate(g, x_std_input[start:end], z_dim=z_dim, J=J)
        fake_y[:, start:end] = chunk * y_std + y_mean

    pred_mean = fake_y.mean(dim=0)
    pred_quantiles = torch.quantile(fake_y, quantiles, dim=0)
    lower, upper = pred_quantiles[0], pred_quantiles[1]

    lpi = (upper - lower).mean()
    cp = ((y_true >= lower) & (y_true <= upper)).float().mean()
    sd_ube = ((upper - y_true) ** 2).mean().sqrt()
    sd_lbe = ((lower - y_true) ** 2).mean().sqrt()

    return {
        'mse': nn.functional.mse_loss(pred_mean, y_true).item(),
        'l1': nn.functional.l1_loss(pred_mean, y_true).item(),
        'lpi': lpi.item(),
        'cp': cp.item(),
        'sd_ube': sd_ube.item(),
        'sd_lbe': sd_lbe.item(),
    }
