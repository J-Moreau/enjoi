r"""Helpers"""

import json
import math
import os
import random
from pathlib import Path
from typing import *

import gdown
import h5py
import numpy as np
import ot
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from tqdm import trange

from ..score import *
from .sevir_dataloader import load_sevir_dataset

DATASET_TO_URL = {'KS': {'train.h5': 'https://drive.google.com/uc?id=1V12ha_CEPDx1dOwn4XNkoOIwxOK3_Tb3',
                         'valid.h5': 'https://drive.google.com/uc?id=1ObDa5gzyJXZr-lTQND6QtVNygcENuSeI',
                         'test.h5': 'https://drive.google.com/uc?id=1_oeI9Ao4KEl84uQAzU7rRyyQK_2cz3v6'
                         },
                  'burgers': {'train.h5': 'https://drive.google.com/uc?id=1G8-By7DySaap8gx9crlG5siFnZ8MvrQH',
                              'valid.h5': 'https://drive.google.com/uc?id=1uI-vvKLjJqI8Az6OJOPRGNdFG7SQDqBr',
                              'test.h5': 'https://drive.google.com/uc?id=12MYFMoYCk3U8t2FmttkdYBS1ZvcJe3o3'
                              },
                }


def random_config(configs: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
    return {key: random.choice(values) for key, values in configs.items()}


def save_config(config: Dict[str, Any], path: Path) -> None:
    with open(path / "config.json", mode="x") as f:
        json.dump(config, f)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path / "config.json", mode="r") as f:
        return json.load(f)


def save_data(x: Tensor, file: Path) -> None:
    with h5py.File(file, mode="w") as f:
        f.create_dataset("x", data=x, dtype=np.float32)


def load_dataset(file: Path, **kwargs) -> dict:
    file_str = str(file)
    file_path = Path(file_str)
    if file_path.is_dir() and ("sevir" in file_path.parts):
        return load_sevir_dataset(file_path, **kwargs)

    if not os.path.exists(file_str):
        os.makedirs('/'.join(file_str.split('/')[:-1]), exist_ok=True)
        dataset_name, split = file_str.split('/')[-2:]
        url = DATASET_TO_URL[dataset_name][split]
        gdown.download(url, file_str, quiet=False)

    with h5py.File(file_str, mode="r") as f:
        data = f["x"][:]

    return {'data': torch.from_numpy(data).float()}


def _unfold_data(data_dict: dict, window: int, spatial: int) -> dict:
    """Apply sliding-window unfolding to a data dict in-place."""
    if window is None:
        pass
    elif window == 1:
        data_dict['data'] = data_dict['data'].flatten(0, 1)
    elif spatial == 1:
        # the method below works strangely when we have 1d data
        # NOTE: the assumption that the SDA paper was doing is that
        # event_shape becomes the channel dimension. We consider it differently
        data = data_dict['data'].unfold(1, window, 1)
        data = data.movedim(2, 3)
        data = data.flatten(0, 1)
        data_dict['data'] = data
    else:
        data = data_dict['data'].unfold(1, window, 1)
        data = data.movedim(-1, 2)
        data = data.flatten(2, 3)
        data = data.flatten(0, 1)
        data_dict['data'] = data
    return data_dict


def load_data(file: Path,
              window: int = None,
              spatial: int = 2) -> Tensor:
    """
    The window argument prepared the pseudo markov blanket
    used for approxiamting the score. However I think there is something
    strange going on here. Or I am just confused.

    NOTE: If we condier a specific window size, shouldn't we be
    symmetric. So I have the feeling that data = data.unfold(1, window, 1)
    should be data = data.unfold(1, 2*window+1, 1). But in this case it's not the case.

    NOTE (2): windows should be an odd number only, otherwise it is not running. Or at least
                this is what is happening in the lorenz experiment. So k = window // 2

    """
    return _unfold_data(load_dataset(file), window, spatial)


def load_zarr_dataset(path: str, split: str, zarr_subpath: str,
                      scale_min: float, scale_max: float) -> dict:
    """Load a zarr array and apply MinMax scaling to [-1, 1]."""
    import zarr
    store_path = os.path.join(path, zarr_subpath.format(split=split))
    arr = zarr.open(store_path, mode="r")
    data = np.array(arr)
    data = 2.0 * (data - scale_min) / (scale_max - scale_min) - 1.0
    return {'data': torch.from_numpy(data).float()}


def load_zarr_data(path: str, split: str, zarr_subpath: str,
                   scale_min: float, scale_max: float,
                   window: int = None, spatial: int = 1) -> dict:
    """Load zarr data and apply windowing (same logic as load_data)."""
    return _unfold_data(load_zarr_dataset(path, split, zarr_subpath, scale_min, scale_max), window, spatial)


def minmaxscale(trainset, validset, testset):
    # Train
    scaler = MinMaxScaler(feature_range = (-1, 1))
    trainset_shape = trainset.shape
    trainset = scaler.fit_transform(trainset.flatten().reshape(-1, 1))
    trainset = torch.from_numpy(trainset.reshape(trainset_shape)).to(torch.float32)
    # Validation
    validset_shape = validset.shape
    validset = scaler.transform(validset.flatten().reshape(-1, 1))
    validset = torch.from_numpy(validset.reshape(validset_shape)).to(torch.float32)
    # Test
    testset_shape = testset.shape
    testset = scaler.transform(testset.flatten().reshape(-1, 1))
    testset = torch.from_numpy(testset.reshape(testset_shape)).to(torch.float32)
    return trainset, validset, testset, scaler

def standardise(trainset, validset, testset):
    # Train
    trainset_mean = trainset.mean()
    trainset_std = trainset.std()
    trainset = (trainset - trainset_mean)/trainset_std
    # Validation
    validset = (validset - trainset_mean)/trainset_std
    # Test
    testset = (testset - trainset_mean)/trainset_std
    return trainset, validset, testset

def scale_std(trainset, validset, testset, target_std = 1.5):
    # Train
    trainset_std = trainset.std()
    trainset = trainset*target_std/trainset_std
    # Validation
    validset = validset*target_std/trainset_std
    # Test
    testset = testset*target_std/trainset_std
    return trainset, validset, testset


def bpf(
    x: Tensor,  # (M, *)
    y: Tensor,  # (N, *)
    transition: Callable[[Tensor], Tensor],
    likelihood: Callable[[Tensor, Tensor], Tensor],
    step: int = 1,
) -> Tensor:  # (M, N + 1, *)
    r"""Performs bootstrap particle filter (BPF) sampling

    .. math:: p(x_0, x_1, ..., x_n | y_1, ..., y_n)
        = p(x_0) \prod_i p(x_i | x_{i-1}) p(y_i | x_i)

    Wikipedia:
        https://wikipedia.org/wiki/Particle_filter

    Arguments:
        x: A set of initial states :math:`x_0`.
        y: The vector of observations :math:`(y_1, ..., y_n)`.
        transition: The transition function :math:`p(x_i | x_{i-1})`.
        likelihood: The likelihood function :math:`p(y_i | x_i)`.
        step: The number of transitions per observation.
    """

    x = x[:, None]

    for yi in y:
        for _ in range(step):
            xi = transition(x[:, -1])
            x = torch.cat((x, xi[:, None]), dim=1)

        w = likelihood(yi, xi)
        j = torch.multinomial(w, len(w), replacement=True)
        x = x[j]

    return x
