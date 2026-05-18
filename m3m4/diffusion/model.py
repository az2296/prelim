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

    def __init__(self, x_dim=1, y_dim=2, t_dim=64):
        super().__init__()
        self.time_emb = TimeEmbedding(t_dim)

        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim + t_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, y_dim),
        )

    def forward(self, x, y_t, t):
        t_emb = self.time_emb(t)
        inputs = torch.cat([x, y_t, t_emb], dim=1)
        return self.net(inputs)
    
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
