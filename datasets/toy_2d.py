
import numpy as np

import torch


def f(x, label):
    if label == 0:
        return np.exp(np.sin(np.pi * x[:, 0]) + x[:, 1]**2)
    else:
        raise ValueError(f"Unknown label: {label}")

def training_dataset(label):
    data = np.random.rand(1000, 2) * 2 - 1
    targets = f(data, label).reshape(-1, 1)
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))

    return dataset


def validation_dataset(label):
    data = np.random.rand(100, 2) * 2 - 1
    targets = f(data, label).reshape(-1, 1)

    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))

    return dataset


def test_dataset(label):
    data = np.random.rand(100, 2) * 2 - 1
    targets = f(data, label).reshape(-1, 1)

    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))

    return dataset
