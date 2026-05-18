from model import make_denoising_batch, sample_y

import torch
import torch.nn as nn
import math


@torch.no_grad()
def eval_loss(model, data, device, scheduler, J=20):
    x, y = data
    x = x.to(device)
    y = y.to(device)
    loss = 0.0
    for _ in range(J):
        _, y_t, t, noise = make_denoising_batch(x, y, scheduler)
        noise_pred = model(x, y_t, t)
        loss += nn.functional.mse_loss(noise_pred, noise).item()
    return loss / J


@torch.no_grad()
def sample_y_many(model, x, scheduler, y_mean, y_std, J, num_inference_steps):
    x_rep = x.repeat(J, 1)
    y = sample_y(model, x_rep, scheduler, y_mean, y_std, num_inference_steps)
    return y.reshape(J, x.shape[0], -1)


def theoretical_stats(x, model):
    x1 = x[:, 0]
    quantiles = torch.tensor((0.05, 0.25, 0.5, 0.75, 0.95), device=x.device, dtype=x.dtype)

    if model == "M3":
        mean = torch.stack([2 * x1 - 0.5, 2 * x1], dim=1)

        sd1 = math.sqrt(2 * math.pi**2 / 3 - 5 / 16 + 0.4**2)
        sd2 = math.sqrt(2 * math.pi**2 / 3 + 1 / 16 + 0.4**2)
        sd = torch.stack([torch.full_like(x1, sd1), torch.full_like(x1, sd2)], dim=1)

        u = torch.rand(200000, device=x.device, dtype=x.dtype) * 2 * math.pi
        eps = 0.4 * torch.randn(200000, 2, device=x.device, dtype=x.dtype)
        noise = torch.stack([u * torch.sin(2 * u), u * torch.cos(2 * u)], dim=1) + eps
        noise_quantiles = torch.quantile(noise, quantiles, dim=0)
        theoretical_quantiles = 2 * x1.view(1, -1, 1) + noise_quantiles.view(5, 1, 2)

    elif model == "M4":
        mean = torch.stack([x1, x1], dim=1)

        sd_value = math.sqrt(8 / 3 + 0.25**2)
        sd = torch.full((x.shape[0], 2), sd_value, device=x.device, dtype=x.dtype)

        means = torch.tensor([-2.0, 0.0, 2.0], device=x.device, dtype=x.dtype)
        idx = torch.randint(0, 3, (200000, 2), device=x.device)
        noise = means[idx] + 0.25 * torch.randn(200000, 2, device=x.device, dtype=x.dtype)
        noise_quantiles = torch.quantile(noise, quantiles, dim=0)
        theoretical_quantiles = x1.view(1, -1, 1) + noise_quantiles.view(5, 1, 2)

    return mean, sd, theoretical_quantiles


@torch.no_grad()
def eval_test_metrics(
    model,
    data,
    device,
    scheduler,
    y_mean,
    y_std,
    data_model,
    J=1000,
    num_inference_steps=20,
):
    x, y_true = data
    x = x.to(device)
    y_true = y_true.to(device)
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)
    quantiles = torch.tensor((0.05, 0.25, 0.5, 0.75, 0.95), device=device, dtype=x.dtype)

    fake_y = sample_y_many(model, x, scheduler, y_mean, y_std, J, num_inference_steps)

    pred_mean = fake_y.mean(dim=0)
    pred_sd = fake_y.std(dim=0, unbiased=False)
    pred_quantiles = torch.quantile(fake_y, quantiles, dim=0)

    true_mean, true_sd, true_quantiles = theoretical_stats(x, data_model)

    return {
        "mse_y_1": nn.functional.mse_loss(pred_mean[:, 0], y_true[:, 0]).item(),
        "mse_y_2": nn.functional.mse_loss(pred_mean[:, 1], y_true[:, 1]).item(),
        "l1_y_1": nn.functional.l1_loss(pred_mean[:, 0], y_true[:, 0]).item(),
        "l1_y_2": nn.functional.l1_loss(pred_mean[:, 1], y_true[:, 1]).item(),
        "mse_mean_y_1": nn.functional.mse_loss(pred_mean[:, 0], true_mean[:, 0]).item(),
        "mse_mean_y_2": nn.functional.mse_loss(pred_mean[:, 1], true_mean[:, 1]).item(),
        "mse_sd_y_1": nn.functional.mse_loss(pred_sd[:, 0], true_sd[:, 0]).item(),
        "mse_sd_y_2": nn.functional.mse_loss(pred_sd[:, 1], true_sd[:, 1]).item(),
        "quantiles": quantiles.tolist(),
        "mse_quantile_y_1": ((pred_quantiles[:, :, 0] - true_quantiles[:, :, 0]) ** 2).mean(dim=1).tolist(),
        "mse_quantile_y_2": ((pred_quantiles[:, :, 1] - true_quantiles[:, :, 1]) ** 2).mean(dim=1).tolist(),
    }
