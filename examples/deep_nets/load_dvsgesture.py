from __future__ import annotations

from pathlib import Path

import tonic
import torch
from tonic import DiskCachedDataset
from torchvision import transforms as vision_transforms


THIS_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = THIS_DIR.parent
DATA_PATH = EXAMPLES_DIR / "data"
CACHE_PATH = EXAMPLES_DIR / "cache"


def _resize_frames(size: int, antialias: bool = True):
    return vision_transforms.Resize((size, size), antialias=antialias)


def get_dvsgesture_dataset(timestep_size: int = 75000, sensor_width: int = 32, cache: bool = True):
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    input_shape = (2, sensor_width, sensor_width)
    sensor_size = tonic.datasets.DVSGesture.sensor_size

    train_dataset = tonic.datasets.DVSGesture(save_to=str(DATA_PATH), train=True)
    test_dataset = tonic.datasets.DVSGesture(save_to=str(DATA_PATH), train=False)

    frame_transform = tonic.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=timestep_size),
        torch.tensor,
        _resize_frames(sensor_width, antialias=True),
    ])

    if cache:
        train_dataset = DiskCachedDataset(
            train_dataset,
            transform=frame_transform,
            cache_path=str(CACHE_PATH / f"DVSGesture_t{timestep_size}_s{sensor_width}_train"),
        )
        test_dataset = DiskCachedDataset(
            test_dataset,
            transform=frame_transform,
            cache_path=str(CACHE_PATH / f"DVSGesture_t{timestep_size}_s{sensor_width}_test"),
        )
    else:
        train_dataset.transform = frame_transform
        test_dataset.transform = frame_transform

    n_outputs = 11
    return train_dataset, test_dataset, input_shape, n_outputs, None
