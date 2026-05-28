from model import make_denoising_batch, sample_y

import torch
import torch.nn as nn


COORD_NAMES = ("lon", "lat", "floor", "buildingid", "spaceid")


@torch.no_grad()
def eval_loss(model, data, device, scheduler, x_mean, x_std, J=5):
    x, y = data
    x = x.to(device)
    y = y.to(device)
    x = (x - x_mean.to(device)) / x_std.to(device)
    loss = 0.0
    for _ in range(J):
        _, y_t, t, noise = make_denoising_batch(x, y, scheduler)
        noise_pred = model(x, y_t, t)
        loss += nn.functional.mse_loss(noise_pred, noise).item()
    return loss / J


@torch.no_grad()
def sample_y_many(model, x, scheduler, y_mean, y_std, J, num_inference_steps, chunk_size=500):
    n = x.shape[0]
    y_dim = y_mean.shape[0]
    out = torch.empty(J, n, y_dim, device=x.device, dtype=x.dtype)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_rep = x[start:end].repeat(J, 1)
        y = sample_y(model, x_rep, scheduler, y_mean, y_std, num_inference_steps)
        out[:, start:end] = y.reshape(J, end - start, y_dim)
    return out


@torch.no_grad()
def eval_test_metrics(
    model,
    data,
    device,
    scheduler,
    x_mean,
    x_std,
    y_mean,
    y_std,
    J=1000,
    num_inference_steps=50,
):
    x, y_true = data
    x = x.to(device)
    y_true = y_true.to(device)
    x_mean = x_mean.to(device)
    x_std = x_std.to(device)
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)
    quantiles = torch.tensor((0.025, 0.975), device=device, dtype=x.dtype)

    x_std_input = (x - x_mean) / x_std
    fake_y = sample_y_many(model, x_std_input, scheduler, y_mean, y_std, J, num_inference_steps)
    fake_y = (fake_y - y_mean) / y_std
    y_true = (y_true - y_mean) / y_std

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
