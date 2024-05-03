import torch as th
from typing import Union
import os
import shutil
from torchvision.ops import focal_loss
from torchvision.transforms import ToPILImage
from PIL import Image
from torch import Tensor
import torch
import numpy as np
import imageio

from torch import nn

ParallelType = Union[nn.DataParallel, nn.parallel.DistributedDataParallel]


class WithStateDict(nn.Module):
    """Wrapper to provide a `state_dict` method to a single tensor."""

    def __init__(self, **tensors):
        super().__init__()
        for name, value in tensors.items():
            self.register_buffer(name, value)
        # self.tensor = nn.Parameter(tensor, requires_grad=False)


def expanduservars(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def archive_code(path: str) -> None:
    shutil.copy("params.yml", path)
    # Copy the current code to the output folder.
    os.system(f"git ls-files -z | xargs -0 tar -czf {os.path.join(path, 'code.tar.gz')}")