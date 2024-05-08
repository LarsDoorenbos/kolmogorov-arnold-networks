
import numpy as np

import torch


def f(x, label):
    if label == 0:
        return np.sin(2 * np.pi * x)
    elif label == 1:
        # Get first decimal digit
        decimals = (x * 10).astype(int) % 10
        decimals = decimals.flatten()

        # To one-hot
        targets = np.zeros((x.shape[0], 10))
        targets[np.arange(x.shape[0]), decimals] = 1

        return targets
    else:
        raise ValueError(f"Unknown label: {label}")

def training_dataset(label):
    data = np.random.rand(1000, 1)
    targets = f(data, label)

    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))

    return dataset


def validation_dataset(label):
    data = np.random.rand(100, 1)
    targets = f(data, label)

    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))

    return dataset


def test_dataset(label):
    data = np.random.rand(100, 1)
    targets = f(data, label)

    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))

    return dataset
