
import numpy as np

import torch


def f(x):
    return np.sin(2 * np.pi * x)

def training_dataset():
    data = np.random.rand(1000, 1)
    targets = f(data)

    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))

    return dataset


def validation_dataset():
    data = np.random.rand(100, 1)
    targets = f(data)

    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))

    return dataset


def test_dataset():
    data = np.random.rand(100, 1)
    targets = f(data)

    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))

    return dataset
