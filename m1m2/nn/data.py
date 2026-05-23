import torch


def sample_m1(rng, n=7000, d=5):
    X = torch.randn(n, d, generator=rng)
    eps = torch.randn(n, generator=rng)
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    mean = x1**2 + torch.exp(x2 + x3 / 3) + x4 - x5
    sigma = 0.5 + x2**2 / 2 + x5**2 / 2
    y = (mean + sigma * eps).unsqueeze(1)
    return X, y


def sample_m2(rng, n=7000, d=5):
    X = torch.randn(n, d, generator=rng)
    eps = torch.randn(n, generator=rng)
    beta = torch.zeros(d)
    beta[:5] = torch.tensor([1., 1., -1., -1., 1.])
    z = X @ beta
    y = (z**2 + torch.sin(z.abs()) + 2 * torch.cos(eps)).unsqueeze(1)
    return X, y


def generate_samples(model='M1', d=5, seed=None, num_train=5000, num_val=1000, num_test=1000):
    n = num_train + num_val + num_test

    rng = torch.Generator()
    rng.manual_seed(seed if seed is not None else torch.seed())

    match model:
        case 'M1':
            X, y = sample_m1(n=n, d=d, rng=rng)
        case 'M2':
            X, y = sample_m2(n=n, d=d, rng=rng)

    train = (X[:num_train], y[:num_train])

    val = (X[num_train:num_train + num_val], y[num_train:num_train + num_val])

    test = (X[num_train + num_val:], y[num_train + num_val:])

    return train, val, test
