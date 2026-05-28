import os

import pandas as pd
import torch


CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "UJIndoorLocData.csv")

X_COLS = [f"WAP{i:03d}" for i in range(1, 521)]
Y_COLS = ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID"]


def load_uji(rng, csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    X = torch.tensor(df[X_COLS].to_numpy(), dtype=torch.float32)
    y = torch.tensor(df[Y_COLS].to_numpy(), dtype=torch.float32)
    perm = torch.randperm(X.shape[0], generator=rng)
    return X[perm], y[perm]


def generate_samples(seed=None, num_train=14948, num_val=1100, num_test=5000):
    rng = torch.Generator()
    rng.manual_seed(seed if seed is not None else torch.seed())

    X, y = load_uji(rng=rng)

    train = (X[:num_train], y[:num_train])
    val = (X[num_train:num_train + num_val], y[num_train:num_train + num_val])
    test = (X[num_train + num_val:num_train + num_val + num_test], y[num_train + num_val:num_train + num_val + num_test])

    return train, val, test
