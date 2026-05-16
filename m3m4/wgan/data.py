import torch

def sample_m3(rng, n = 7000):
    x   = torch.randn(n, 1, generator=rng)                     
    u   = torch.rand(n, generator=rng) * 2 * torch.pi         
    eps1 = 0.4 * torch.randn(n, generator=rng)
    eps2 = 0.4 * torch.randn(n, generator=rng)
    y1 = 2*x[:,0] + u*torch.sin(2*u) + eps1
    y2 = 2*x[:,0] + u*torch.cos(2*u) + eps2
    y  = torch.stack([y1, y2], dim=1)                          
    return x, y


def sample_m4(rng, n = 7000):
    def gmm_noise():
        means = torch.tensor([-2., 0., 2.])
        idx   = torch.randint(0, 3, (n,), generator=rng)
        mu    = means[idx]
        return mu + 0.25 * torch.randn(n, generator=rng)

    x    = torch.randn(n, 1, generator=rng)
    y1   = x[:,0] + gmm_noise()
    y2   = x[:,0] + gmm_noise()
    y    = torch.stack([y1, y2], dim=1)
    return x, y

def generate_samples(model = 'M3',seed = None, num_train = 5000, num_val = 1000, num_test = 1000):
    n = num_train + num_val + num_test

    rng = torch.Generator()
    rng.manual_seed(seed if seed is not None else torch.seed())
    
    match model:
        case 'M3': X, y = sample_m3(rng = rng, n = n)
        case 'M4': X, y = sample_m4(rng = rng, n = n)
            
    train = (X[:num_train], y[:num_train])

    val = (X[num_train:num_train + num_val], y[num_train:num_train + num_val])

    test = (X[num_train + num_val:], y[num_train + num_val:])
    
    return train, val, test

