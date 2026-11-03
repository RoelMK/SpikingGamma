from spikinggamma.utils.datasets.common import TimeSliceTransform
import torch
import tonic
from tonic import transforms, DiskCachedDataset

DVSGESTURE_SAMPLE_LENGTH_US = 6000000

def get_dvsgesture_dataset(data_path, cache_path, timestep_size = 10000, sensor_width=32, cache=True):
    data_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)
    input_shape = (2, sensor_width, sensor_width)
    transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size, target_size=(sensor_width, sensor_width)),
        transforms.ToFrame(sensor_size=(sensor_width, sensor_width, 2), time_window=timestep_size),
        torch.from_numpy,
        TimeSliceTransform(time_steps=DVSGESTURE_SAMPLE_LENGTH_US // timestep_size)
    ])
    train_dataset = tonic.datasets.DVSGesture(save_to=str(data_path), transform=transform, train=True)
    test_dataset = tonic.datasets.DVSGesture(save_to=str(data_path), transform=transform, train=False)
    if cache:
        train_cache_path = cache_path / f"DVSGesture_t{timestep_size}_s{sensor_width}_train"
        test_cache_path = cache_path / f"DVSGesture_t{timestep_size}_s{sensor_width}_test"
        train_dataset = DiskCachedDataset(train_dataset, transform=torch.from_numpy, cache_path=str(train_cache_path))
        test_dataset = DiskCachedDataset(test_dataset, transform=torch.from_numpy, cache_path=str(test_cache_path))
    n_outputs = 11
    T = DVSGESTURE_SAMPLE_LENGTH_US // timestep_size
    return train_dataset, test_dataset, input_shape, n_outputs, T

