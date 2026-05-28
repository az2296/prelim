from model import mc_generate

import torch
import torch.nn as nn


COORD_NAMES = ("lon", "lat", "floor", "buildingid", "spaceid")


@torch.no_grad()
def eval_mse(g, data, device, x_mean, x_std, y_mean, y_std, z_dim=100, J=20):
    x, y_true = data
    x = x.to(device)
    y_true = (y_true.to(device) - y_mean.to(device)) / y_std.to(device)

    x_std_input = (x - x_mean.to(device)) / x_std.to(device)
    fake_y_std = mc_generate(g, x_std_input, z_dim=z_dim, J=J)
    pred_y = fake_y_std.mean(dim=0)

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
    y_true = y_true.to(device)
    x_mean = x_mean.to(device)
    x_std = x_std.to(device)
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)
    quantiles = torch.tensor((0.025, 0.975), device=device, dtype=x.dtype)

    y_true = (y_true - y_mean) / y_std
    x_std_input = (x - x_mean) / x_std

    chunk_size = 500
    n = x_std_input.shape[0]
    y_dim = y_true.shape[1]
    fake_y = torch.empty(J, n, y_dim, device=device, dtype=x.dtype)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        fake_y[:, start:end] = mc_generate(g, x_std_input[start:end], z_dim=z_dim, J=J)

    pred_mean = fake_y.mean(dim=0)
    pred_quantiles = torch.quantile(fake_y, quantiles, dim=0)
    lower, upper = pred_quantiles[0], pred_quantiles[1]

    metrics = {}
    for i, name in enumerate(COORD_NAMES):
        metrics[f"mse_{name}"] = nn.functional.mse_loss(pred_mean[:, i], y_true[:, i]).item()
        metrics[f"l1_{name}"] = nn.functional.l1_loss(pred_mean[:, i], y_true[:, i]).item()
        metrics[f"cp_{name}"] = ((y_true[:, i] >= lower[:, i]) & (y_true[:, i] <= upper[:, i])).float().mean().item()
        metrics[f"lpi_{name}"] = (upper[:, i] - lower[:, i]).mean().item()

    return metrics
