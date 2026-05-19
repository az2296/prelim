import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler


def make_scheduler(train_steps):
    return DDIMScheduler(num_train_timesteps=train_steps, clip_sample=False)


def make_noisy_batch(x, y, scheduler):
    images = x[:, :1].clone()
    images[:, :, 7:21, 7:21] = y

    masked = x[:, :1]
    mask = x[:, 1:2]

    noise = torch.randn_like(images)
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (images.shape[0],),
        device=images.device,
    )
    noisy = scheduler.add_noise(images, noise, timesteps)

    denoiser_input = torch.cat([noisy, masked, mask], dim=1)
    t = timesteps.float() / scheduler.config.num_train_timesteps

    return denoiser_input, noise, t


@torch.no_grad()
def sample(model, x, scheduler, steps=50, eta=0.0):
    model.eval()

    masked = x[:, :1]
    mask = x[:, 1:2]
    image = torch.randn_like(masked)
    known_noise = torch.randn_like(masked)

    scheduler.set_timesteps(steps, device=x.device)

    for timestep in scheduler.timesteps:
        timesteps = torch.full((x.shape[0],), timestep, device=x.device, dtype=torch.long)
        noisy_known = scheduler.add_noise(
            masked,
            known_noise,
            timesteps,
        )
        image = image * mask + noisy_known * (1 - mask)

        t = timesteps.float() / scheduler.config.num_train_timesteps
        denoiser_input = torch.cat([image, masked, mask], dim=1)
        noise_pred = model(denoiser_input, t)
        image = scheduler.step(noise_pred, timestep, image, eta=eta).prev_sample

    return image * mask + masked * (1 - mask)


class MuonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight_2d = nn.Parameter(torch.empty(out_channels, in_channels * kernel_size * kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound = 1 / math.sqrt(fan_in)

        nn.init.kaiming_uniform_(self.weight_2d, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight = self.weight_2d.reshape(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding)


class MuonConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight_2d = nn.Parameter(torch.empty(in_channels, out_channels * kernel_size * kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound = 1 / math.sqrt(fan_in)

        nn.init.kaiming_uniform_(self.weight_2d, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight = self.weight_2d.reshape(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
        )
        return F.conv_transpose2d(x, weight, self.bias, stride=self.stride, padding=self.padding)


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

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, groups = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = MuonConv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = MuonConv2d(out_ch, out_ch, 3, padding=1)
        self.skip = MuonConv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Denoiser(nn.Module):

    def __init__(self, base_channels = 64, t_dim = 128):
        super().__init__()
        b = base_channels
        self.time_mlp = TimeEmbedding(t_dim)
        self.in_conv = MuonConv2d(3, b, 3, padding=1)

        self.rb1 = ResBlock(b, b, t_dim)
        self.down1 = MuonConv2d(b, b * 2, 4, stride=2, padding=1)       
        self.rb2 = ResBlock(b * 2, b * 2, t_dim)
        self.down2 = MuonConv2d(b * 2, b * 4, 4, stride=2, padding=1)   
        self.rb3 = ResBlock(b * 4, b * 4, t_dim)

        self.mid1 = ResBlock(b * 4, b * 4, t_dim)
        self.mid2 = ResBlock(b * 4, b * 4, t_dim)

        self.up1 = MuonConvTranspose2d(b * 4, b * 2, 4, stride=2, padding=1)
        self.rb4 = ResBlock(b * 4, b * 2, t_dim)
        self.up2 = MuonConvTranspose2d(b * 2, b, 4, stride=2, padding=1)
        self.rb5 = ResBlock(b * 2, b, t_dim)

        self.out_norm = nn.GroupNorm(8, b)
        self.out_conv = MuonConv2d(b, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h1 = self.rb1(self.in_conv(x), t_emb)
        h2 = self.rb2(self.down1(h1), t_emb)
        h3 = self.rb3(self.down2(h2), t_emb)
        h = self.mid2(self.mid1(h3, t_emb), t_emb)
        h = self.rb4(torch.cat([self.up1(h), h2], dim=1), t_emb)
        h = self.rb5(torch.cat([self.up2(h), h1], dim=1), t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))
