from __future__ import annotations

from pathlib import Path

import tonic
import torch
from tonic import DiskCachedDataset, transforms


DVSGESTURE_SAMPLE_LENGTH = 6_000_000

THIS_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = THIS_DIR.parent
DATA_PATH = EXAMPLES_DIR / "data"
CACHE_PATH = EXAMPLES_DIR / "cache"


class TimeSliceTransform:
    def __init__(self, time_steps: int, time_dim: int = 0):
        self.time_steps = time_steps
        self.time_dim = time_dim

    def __call__(self, x: torch.Tensor):
        tdim = self.time_dim
        size = x.shape[tdim]
        if size > self.time_steps:
            return x.narrow(tdim, 0, self.time_steps)
        if size < self.time_steps:
            pad_shape = list(x.shape)
            pad_shape[tdim] = self.time_steps - size
            pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=tdim)
        return x


def get_dvsgesture_dataset(
    timestep_size: int = 10000,
    sensor_width: int = 32,
    cache: bool = True,
    data_path: str | Path = DATA_PATH,
    cache_path: str | Path = CACHE_PATH,
):
    data_path = Path(data_path)
    cache_path = Path(cache_path)
    data_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    input_shape = (2, sensor_width, sensor_width)
    transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        tonic.transforms.Downsample(
            sensor_size=tonic.datasets.DVSGesture.sensor_size,
            target_size=(sensor_width, sensor_width),
        ),
        transforms.ToFrame(
            sensor_size=(sensor_width, sensor_width, 2),
            time_window=timestep_size,
        ),
        torch.from_numpy,
        TimeSliceTransform(time_steps=DVSGESTURE_SAMPLE_LENGTH // timestep_size),
    ])

    train_dataset = tonic.datasets.DVSGesture(
        save_to=str(data_path),
        transform=transform,
        train=True,
    )
    test_dataset = tonic.datasets.DVSGesture(
        save_to=str(data_path),
        transform=transform,
        train=False,
    )

    if cache:
        train_cache_path = cache_path / f"DVSGesture_t{timestep_size}_s{sensor_width}_train"
        test_cache_path = cache_path / f"DVSGesture_t{timestep_size}_s{sensor_width}_test"
        train_dataset = DiskCachedDataset(
            train_dataset,
            transform=torch.from_numpy,
            cache_path=str(train_cache_path),
        )
        test_dataset = DiskCachedDataset(
            test_dataset,
            transform=torch.from_numpy,
            cache_path=str(test_cache_path),
        )

    n_outputs = 11
    time_steps = DVSGESTURE_SAMPLE_LENGTH // timestep_size
    return train_dataset, test_dataset, input_shape, n_outputs, time_steps
