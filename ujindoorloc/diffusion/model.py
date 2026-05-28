import torch
import math
import torch.nn as nn
from diffusers import DDIMScheduler

class TimeEmbedding(nn.Module):

    def __init__(self, t_dim=64):
        super().__init__()
        self.t_dim = t_dim

        self.net = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

    def forward(self, t):

        half_dim = self.t_dim // 2

        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device)
            * math.log(10000.0)
            / half_dim
        )

        args = 1_000 * t.unsqueeze(1) * freqs.unsqueeze(0)

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        return self.net(emb)

class Denoiser(nn.Module):

    def __init__(self, x_dim=520, y_dim=5, t_dim=64):
        super().__init__()
        self.time_emb = TimeEmbedding(t_dim)

        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim + t_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, y_dim),
        )

    def forward(self, x, y_t, t):
        t_emb = self.time_emb(t)
        inputs = torch.cat([x, y_t, t_emb], dim=1)
        return self.net(inputs)

    def muon_param_groups(self):
        muon, adamw = [], []
        for name, p in self.named_parameters():
            if name == "net.2.weight":
                muon.append(p)
            else:
                adamw.append(p)
        return muon, adamw


class FiLMBlock(nn.Module):

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        self.act = nn.SiLU()

    def forward(self, h, cond):
        gamma, beta = self.film(cond).chunk(2, dim=1)
        x = self.norm(h)
        x = self.act(self.lin1(x))
        x = x * (1 + gamma) + beta
        x = self.act(self.lin2(x))
        return h + x


class FiLMDenoiser(nn.Module):

    def __init__(self, x_dim=520, y_dim=5, t_dim=64, hidden=128, n_blocks=4):
        super().__init__()
        self.time_emb = TimeEmbedding(t_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(x_dim + t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.in_proj = nn.Linear(y_dim, hidden)
        self.blocks = nn.ModuleList(FiLMBlock(hidden, hidden) for _ in range(n_blocks))
        self.out = nn.Linear(hidden, y_dim)

    def forward(self, x, y_t, t):
        t_emb = self.time_emb(t)
        cond = self.cond_mlp(torch.cat([x, t_emb], dim=1))
        h = self.in_proj(y_t)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(h)

    def muon_param_groups(self):
        muon_names = {"cond_mlp.2.weight"}
        for i in range(len(self.blocks)):
            for w in ("lin1.weight", "lin2.weight", "film.weight"):
                muon_names.add(f"blocks.{i}.{w}")

        muon, adamw = [], []
        for name, p in self.named_parameters():
            if name in muon_names:
                muon.append(p)
            else:
                adamw.append(p)
        return muon, adamw


def make_denoiser(arch="mlp", t_dim=64):
    if arch == "film":
        return FiLMDenoiser(t_dim=t_dim)
    return Denoiser(t_dim=t_dim)


def update_ema(model, ema_model, decay=0.999):
    with torch.no_grad():
        for p, p_ema in zip(model.parameters(), ema_model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)


def make_scheduler(num_train_timesteps=1000):
    return DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )


def scale_t(timesteps, scheduler):
    return timesteps.float() / (scheduler.config.num_train_timesteps - 1)


def make_denoising_batch(x, y0, scheduler):
    batch_size = x.shape[0]
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (batch_size,),
        device=x.device,
    )
    noise = torch.randn_like(y0)
    y_t = scheduler.add_noise(y0, noise, timesteps)
    t = scale_t(timesteps, scheduler)
    return x, y_t, t, noise


@torch.no_grad()
def sample_y(model, x, scheduler, y_mean, y_std, num_inference_steps=50):

    scheduler.set_timesteps(num_inference_steps, device=x.device)

    y = torch.randn(x.shape[0], y_mean.shape[0], device=x.device, dtype=x.dtype)

    for timestep in scheduler.timesteps:
        timesteps = timestep.expand(x.shape[0]).to(x.device)
        t = scale_t(timesteps, scheduler).to(x.dtype)
        noise_pred = model(x, y, t)
        y = scheduler.step(noise_pred, timestep, y).prev_sample

    return y*y_std + y_mean
