import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MuonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight_2d = nn.Parameter(
            torch.empty(out_channels, in_channels * kernel_size * kernel_size)
        )
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

        self.weight_2d = nn.Parameter(
            torch.empty(in_channels, out_channels * kernel_size * kernel_size)
        )
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


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, slope=0.2, groups=8):
        super().__init__()
        self.slope = slope
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = MuonConv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = MuonConv2d(out_ch, out_ch, 3, padding=1)
        self.skip = MuonConv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.leaky_relu(self.norm1(x), self.slope))
        h = self.conv2(F.leaky_relu(self.norm2(h), self.slope))
        return h + self.skip(x)

 
class Generator(nn.Module):
    def __init__(self, z_dim=100, base_channels=64, slope=0.2):
        super().__init__()
        b = base_channels
        self.in_conv = MuonConv2d(2, b, 3, padding=1)
        self.rb1 = ResBlock(b, b, slope)
        self.down1 = MuonConv2d(b, b * 2, 4, stride=2, padding=1)
        self.rb2 = ResBlock(b * 2, b * 2, slope)
        self.down2 = MuonConv2d(b * 2, b * 4, 4, stride=2, padding=1)
        self.rb3 = ResBlock(b * 4, b * 4, slope)
        self.z_proj = nn.Linear(z_dim, b * 4 * 7 * 7)
        self.mid1 = ResBlock(b * 4, b * 4, slope)
        self.mid2 = ResBlock(b * 4, b * 4, slope)
        self.up1 = MuonConvTranspose2d(b * 4, b * 2, 4, stride=2, padding=1)
        self.rb4 = ResBlock(b * 4, b * 2, slope)
        self.up2 = MuonConvTranspose2d(b * 2, b, 4, stride=2, padding=1)
        self.rb5 = ResBlock(b * 2, b, slope)
        self.out_conv = MuonConv2d(b, 1, 3, padding=1)

    def forward(self, masked, mask, z):
        batch_size = masked.size(0)
        cond = torch.cat([masked, mask], dim=1)
        h1 = self.rb1(self.in_conv(cond))
        h2 = self.rb2(self.down1(h1))
        h3 = self.rb3(self.down2(h2))
        h = h3 + self.z_proj(z).view(batch_size, h3.size(1), h3.size(2), h3.size(3))
        h = self.mid2(self.mid1(h))
        h = self.rb4(torch.cat([self.up1(h), h2], dim=1))
        h = self.rb5(torch.cat([self.up2(h), h1], dim=1))
        generated = torch.tanh(self.out_conv(h))
        return (1.0 - mask) * masked + mask * generated

    def muon_param_groups(self):
        muon, adamw = [], []
        for name, p in self.named_parameters():
            if "weight_2d" in name and "in_conv" not in name and "out_conv" not in name:
                muon.append(p)
            else:
                adamw.append(p)
        return muon, adamw


class Critic(nn.Module):
    def __init__(self, base_channels=64, slope=0.2):
        super().__init__()
        b = base_channels
        self.net = nn.Sequential(
            MuonConv2d(3, b, 3, padding=1),
            nn.LeakyReLU(slope),
            MuonConv2d(b, b * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(slope),
            MuonConv2d(b * 2, b * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(slope),
            MuonConv2d(b * 4, b * 4, 3, padding=1),
            nn.LeakyReLU(slope),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(b * 4 * 7 * 7, b * 4),
            nn.LeakyReLU(slope),
            nn.Linear(b * 4, 1),
        )

    def forward(self, masked, mask, image):
        x = torch.cat([image, masked, mask], dim=1)
        return self.head(self.net(x)).view(-1)

    def muon_param_groups(self):
       
        muon, adamw = [], []
        for name, p in self.named_parameters():
            is_hidden_conv = "weight_2d" in name and "net.0" not in name
            is_hidden_linear = name == "head.1.weight"
            if is_hidden_conv or is_hidden_linear:
                muon.append(p)
            else:
                adamw.append(p)
        return muon, adamw


def gradient_penalty(critic, masked, mask, real, fake, lambda_gp):
    batch_size = real.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)

    masked_interp = masked.detach().requires_grad_(True)
    image_interp = (alpha * real + (1 - alpha) * fake).detach().requires_grad_(True)
    score = critic(masked_interp, mask, image_interp)

    grad_masked, grad_image = torch.autograd.grad(
        score,
        [masked_interp, image_interp],
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    grad = torch.cat([grad_masked.flatten(1), grad_image.flatten(1)], dim=1)
    return lambda_gp * (grad.norm(2, dim=1) - 1).pow(2).mean()
