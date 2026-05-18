from model import make_denoising_batch, sample_y

import torch
import torch.nn as nn
import math


@torch.no_grad()
def eval_loss(model, data, device, scheduler, J=5):
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
    return y.reshape(J, x.shape[0])


def theoretical_stats(x, model):
    x1, x2, x3, x4, x5 = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4]
    quantiles = torch.tensor((0.05, 0.25, 0.5, 0.75, 0.95), device=x.device, dtype=x.dtype)

    if model == 'M1':
        mean = x1**2 + torch.exp(x2 + x3/3) + x4 - x5
        sd = 0.5 + x2**2/2 + x5**2/2

        normal = torch.distributions.Normal(
            torch.tensor(0.0, device=x.device, dtype=x.dtype),
            torch.tensor(1.0, device=x.device, dtype=x.dtype)
        )
        theoretical_quantiles = (
            mean.unsqueeze(0)
            + sd.unsqueeze(0) * normal.icdf(quantiles).unsqueeze(1)
        )

    elif model == 'M2':
        beta = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        beta[:5] = torch.tensor([1., 1., -1., -1., 1.], device=x.device, dtype=x.dtype)
        z = x @ beta
        base = z**2 + torch.sin(z.abs())

        mean = base + 2 * math.exp(-0.5)
        var_cos = (1 + math.exp(-2)) / 2 - math.exp(-1)
        sd = torch.full_like(mean, 2 * math.sqrt(var_cos))

        normal = torch.distributions.Normal(
            torch.tensor(0.0, device=x.device, dtype=x.dtype),
            torch.tensor(1.0, device=x.device, dtype=x.dtype)
        )
        u = torch.linspace(1e-6, 1 - 1e-6, 200000, device=x.device, dtype=x.dtype)
        noise_samples = 2 * torch.cos(normal.icdf(u))
        noise_quantiles = torch.quantile(noise_samples, quantiles)
        theoretical_quantiles = base.unsqueeze(0) + noise_quantiles.unsqueeze(1)

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
    num_inference_steps=50,
):
    x, y_true = data
    x = x.to(device)
    y_true = y_true.to(device).squeeze(1)
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)
    quantiles = torch.tensor((0.05, 0.25, 0.5, 0.75, 0.95), device=device, dtype=x.dtype)

    fake_y = sample_y_many(model, x, scheduler, y_mean, y_std, J, num_inference_steps)

    pred_mean = fake_y.mean(dim=0)
    pred_sd = fake_y.std(dim=0, unbiased=False)
    pred_quantiles = torch.quantile(fake_y, quantiles, dim=0)

    true_mean, true_sd, true_quantiles = theoretical_stats(x, data_model)
    mse_quantiles = ((pred_quantiles - true_quantiles) ** 2).mean(dim=1)

    return {
        'mse': nn.functional.mse_loss(pred_mean, y_true).item(),
        'l1': nn.functional.l1_loss(pred_mean, y_true).item(),
        'mse_mean': nn.functional.mse_loss(pred_mean, true_mean).item(),
        'mse_sd': nn.functional.mse_loss(pred_sd, true_sd).item(),
        'quantiles': quantiles.tolist(),
        'mse_quantile': mse_quantiles.tolist(),
    }
