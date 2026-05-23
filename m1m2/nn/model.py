import torch
import torch.nn as nn


class Regressor(nn.Module):

    def __init__(self, x_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

    def muon_param_groups(self):
        muon, adamw = [], []
        for name, p in self.named_parameters():
            if name == "net.2.weight":
                muon.append(p)
            else:
                adamw.append(p)
        return muon, adamw


def update_ema(model, ema_model, decay=0.999):
    with torch.no_grad():
        for p, p_ema in zip(model.parameters(), ema_model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)
