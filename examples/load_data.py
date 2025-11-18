import os
from tonic import transforms, DiskCachedDataset
import torch
import tonic

shd_sample_length = 900000
dvsgesture_sample_length = 6000000
data_path = "./data"
cache_path = "./cache"


class TimeSliceTransform:
    def __init__(self, time_steps, time_dim=0):
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


def get_dvsgesture_dataset(timestep_size = 10000, sensor_width=32, cache=True):
    input_shape = (2, sensor_width, sensor_width)
    transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size, target_size=(sensor_width, sensor_width)),
        transforms.ToFrame(sensor_size=(sensor_width, sensor_width, 2), time_window=timestep_size),
        torch.from_numpy,
        TimeSliceTransform(time_steps=dvsgesture_sample_length // timestep_size)
    ])
    train_dataset = tonic.datasets.DVSGesture(save_to=data_path, transform=transform, train=True)
    test_dataset = tonic.datasets.DVSGesture(save_to=data_path, transform=transform, train=False)
    if cache:
        train_cache_path = f"{cache_path}/DVSGesture_t{timestep_size}_s{sensor_width}_train"
        test_cache_path = f"{cache_path}/DVSGesture_t{timestep_size}_s{sensor_width}_test"
        train_dataset = DiskCachedDataset(train_dataset, transform=torch.from_numpy, cache_path=train_cache_path)
        test_dataset = DiskCachedDataset(test_dataset, transform=torch.from_numpy, cache_path=test_cache_path)
    n_outputs = 11
    T = dvsgesture_sample_length // timestep_size
    return train_dataset, test_dataset, input_shape, n_outputs, T


def get_shd_dataset(timestep_size = 3600, sensor_width = 140, cache = True):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    sample_length = shd_sample_length
    T = sample_length // timestep_size

    transform = transforms.Compose([
        transforms.Downsample(spatial_factor=sensor_width / 700.0),
        transforms.ToFrame(sensor_size=(sensor_width, 1, 1), time_window=timestep_size),
        torch.from_numpy,
        TimeSliceTransform(time_steps=T)
    ])

    train_dataset = tonic.datasets.SHD(save_to=data_path, transform=transform, train=True)
    test_dataset = tonic.datasets.SHD(save_to=data_path, transform=transform, train=False)
    if cache:
        train_dataset = DiskCachedDataset(train_dataset, transform=torch.from_numpy, cache_path=f"{cache_path}/SHD_t{timestep_size}_s{sensor_width}_train")
        test_dataset = DiskCachedDataset(test_dataset, transform=torch.from_numpy, cache_path=f"{cache_path}/SHD_t{timestep_size}_s{sensor_width}_test")
    input_shape = (sensor_width, 1, 1)
    n_outputs = 20
    return train_dataset, test_dataset, input_shape, n_outputs, T