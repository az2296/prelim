import torch
import torch.nn as nn
import math


@torch.no_grad()
def predict(model, x, device, y_mean, y_std):
    x = x.to(device)
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)
    pred_y_std = model(x)
    return pred_y_std * y_std + y_mean


@torch.no_grad()
def eval_mse(model, data, device, y_mean, y_std):
    x, y_true = data
    y_true = y_true.to(device)
    pred_y = predict(model, x, device, y_mean, y_std)
    return nn.functional.mse_loss(pred_y, y_true).item()


def theoretical_mean(x, model):
    x1, x2, x3, x4, x5 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

    if model == "M1":
        return x1**2 + torch.exp(x2 + x3 / 3) + x4 - x5

    if model == "M2":
        beta = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        beta[:5] = torch.tensor([1., 1., -1., -1., 1.], device=x.device, dtype=x.dtype)
        z = x @ beta
        return z**2 + torch.sin(z.abs()) + 2 * math.exp(-0.5)

    raise ValueError(f"unknown data model: {model}")


@torch.no_grad()
def eval_test_metrics(model, data, device, y_mean, y_std, data_model):
    x, y_true = data
    x = x.to(device)
    y_true = y_true.to(device).squeeze(1)
    pred_y = predict(model, x, device, y_mean, y_std)
    pred_y = pred_y.squeeze(1)
    true_mean = theoretical_mean(x, data_model)

    return {
        'mse': nn.functional.mse_loss(pred_y, y_true).item(),
        'l1': nn.functional.l1_loss(pred_y, y_true).item(),
        'mse_mean': nn.functional.mse_loss(pred_y, true_mean).item(),
    }
