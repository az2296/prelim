import os

import pandas as pd
import torch


CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CT.csv")


def load_ct(rng, csv_path = CSV_PATH):
    df = pd.read_csv(csv_path)
    x_cols = [f"value{i}" for i in range(384)]
    X = torch.tensor(df[x_cols].to_numpy(), dtype=torch.float32)
    y = torch.tensor(df["reference"].to_numpy(), dtype=torch.float32).unsqueeze(1)
    perm = torch.randperm(X.shape[0], generator=rng)
    return X[perm], y[perm]


def generate_samples(seed = None, num_train = 40000, num_val = 3500, num_test = 10000):
    rng = torch.Generator()
    rng.manual_seed(seed if seed is not None else torch.seed())

    X, y = load_ct(rng = rng)

    train = (X[:num_train], y[:num_train])

    val = (X[num_train:num_train + num_val], y[num_train:num_train + num_val])

    test = (X[num_train + num_val:num_train + num_val + num_test], y[num_train + num_val:num_train + num_val + num_test])

    return train, val, test

