import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, x_dim = 384, z_dim = 100, leaky_slope = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim+z_dim, 128),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(128,64),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(64,1)
        )

    def forward(self, x, z):
        inputs = torch.cat([x,z], dim = 1)
        return self.net(inputs)

class Critic(nn.Module):

    def __init__(self, x_dim = 384, leaky_slope = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim+1, 128),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(128,64),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(64,1)
        )

    def forward(self, x, y):
        inputs = torch.cat([x,y], dim = 1)
        return self.net(inputs)


def c_loss(real_scores, fake_scores):
    return fake_scores.mean() - real_scores.mean()

def g_loss(fake_scores, fake_y, y, recon_weight = 0.1):

    adv_loss = -fake_scores.mean()

    if recon_weight == 0:
        return adv_loss

    mean_fake_y = fake_y.mean(dim=0)

    recon_loss = ((y.squeeze(1)- mean_fake_y) ** 2).mean()

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

    fake_y = g(x_rep, z).view(J,batch_size)

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
