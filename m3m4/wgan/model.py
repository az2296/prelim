import torch
import torch.nn as nn

class Generator(nn.Module):
    
    def __init__(self, z_dim = 100, leaky_slope = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1+z_dim, 512),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(512,512),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(512,512),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(512,2)
        )

    def forward(self, x, z):
        inputs = torch.cat([x,z], dim = 1)
        return self.net(inputs)

    def muon_param_groups(self):
        muon, adamw = [], []
        for name, p in self.named_parameters():
            if name in {"net.2.weight", "net.4.weight"}:
                muon.append(p)
            else:
                adamw.append(p)
        return muon, adamw

class Critic(nn.Module):

    def __init__(self, leaky_slope = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 512),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(512,512),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(512,512),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(512,1)
        )

    def forward(self, x, y):
        inputs = torch.cat([x,y], dim = 1)
        return self.net(inputs)

    def muon_param_groups(self):
        muon, adamw = [], []
        for name, p in self.named_parameters():
            if name in {"net.2.weight", "net.4.weight"}:
                muon.append(p)
            else:
                adamw.append(p)
        return muon, adamw


class GenFiLMBlock(nn.Module):

    def __init__(self, dim, cond_dim, leaky_slope=0.01):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        self.act = nn.LeakyReLU(leaky_slope)

    def forward(self, h, cond):
        gamma, beta = self.film(cond).chunk(2, dim=1)
        x = self.act(self.lin1(h))
        x = x * (1 + gamma) + beta
        x = self.act(self.lin2(x))
        return h + x


class FiLMGenerator(nn.Module):

    def __init__(self, z_dim=100, leaky_slope=0.01, x_dim=1, y_dim=2, hidden=512, n_blocks=4):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(hidden, hidden),
        )
        self.in_proj = nn.Linear(z_dim, hidden)
        self.blocks = nn.ModuleList(
            GenFiLMBlock(hidden, hidden, leaky_slope) for _ in range(n_blocks)
        )
        self.out = nn.Linear(hidden, y_dim)

    def forward(self, x, z):
        cond = self.cond_mlp(x)
        h = self.in_proj(z)
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


class FiLMCritic(nn.Module):

    def __init__(self, leaky_slope=0.01, x_dim=1, y_dim=2, hidden=512, n_blocks=4):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(hidden, hidden),
        )
        self.in_proj = nn.Linear(y_dim, hidden)
        self.blocks = nn.ModuleList(
            GenFiLMBlock(hidden, hidden, leaky_slope) for _ in range(n_blocks)
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, y):
        cond = self.cond_mlp(x)
        h = self.in_proj(y)
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


def make_generator(arch="mlp", z_dim=100, leaky_slope=0.01):
    if arch == "film":
        return FiLMGenerator(z_dim=z_dim, leaky_slope=leaky_slope)
    return Generator(z_dim=z_dim, leaky_slope=leaky_slope)


def make_critic(arch="mlp", leaky_slope=0.01):
    if arch == "film":
        return FiLMCritic(leaky_slope=leaky_slope)
    return Critic(leaky_slope=leaky_slope)


def c_loss(real_scores, fake_scores):
    return fake_scores.mean() - real_scores.mean()

def g_loss(fake_scores, fake_y, y, recon_weight = 0.1):
    
    adv_loss = -fake_scores.mean()

    if recon_weight == 0:
        return adv_loss

    mean_fake_y = fake_y.mean(dim=0)

    recon_loss = ((y - mean_fake_y) ** 2).mean()

    return (1 - recon_weight) * adv_loss + recon_weight * recon_loss

def update_ema(model, ema_model, decay=0.999):
    with torch.no_grad():
        for p, p_ema in zip(model.parameters(), ema_model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)


def mc_generate(g, x, z_dim=100, J=1):
    
    batch_size = x.shape[0]
    device = x.device

    x_rep = x.repeat(J, 1)
    z = torch.randn(J * batch_size, z_dim, device=device)

    fake_y = g(x_rep, z).view(J, batch_size, -1)

    return fake_y

def gradient_penalty(critic, x_real, y_real, x_fake, y_fake, lambda_gp=10.0):
    batch_size = x_real.shape[0]
    alpha = torch.rand(batch_size, 1, device=x_real.device)

    x_interp = (alpha * x_real + (1 - alpha) * x_fake).detach().requires_grad_(True)
    y_interp = (alpha * y_real + (1 - alpha) * y_fake).detach().requires_grad_(True)

    scores = critic(x_interp, y_interp)
    grad_x, grad_y = torch.autograd.grad(
        outputs=scores,
        inputs=[x_interp, y_interp],
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )

    gradients = torch.cat(
        [
            grad_x.reshape(batch_size, -1),
            grad_y.reshape(batch_size, -1)
        ],
        dim=1
    )
    return lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
