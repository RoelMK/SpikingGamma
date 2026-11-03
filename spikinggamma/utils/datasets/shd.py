import torch
import tonic
from tonic import transforms, DiskCachedDataset

from spikinggamma.utils.datasets.common import BinarizeSpikeCounts, TimeSliceTransform

SHD_SAMPLE_LENGTH_US = 900000
SHD_SOURCE_SENSOR_WIDTH = 700
SHD_N_OUTPUTS = 20


def get_shd_dataset(data_path, cache_path, timestep_size = 3600, sensor_width = 140, cache = True, one_spike_per_channel_per_timestep = False):
    data_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    sample_length = SHD_SAMPLE_LENGTH_US
    T = sample_length // timestep_size

    frame_transforms = [
        transforms.Downsample(spatial_factor=sensor_width / SHD_SOURCE_SENSOR_WIDTH),
        transforms.ToFrame(sensor_size=(sensor_width, 1, 1), time_window=timestep_size),
        torch.from_numpy,
    ]
    if one_spike_per_channel_per_timestep:
        frame_transforms.append(BinarizeSpikeCounts())
    frame_transforms.append(TimeSliceTransform(time_steps=T))
    transform = transforms.Compose(frame_transforms)

    train_dataset = tonic.datasets.SHD(save_to=str(data_path), transform=transform, train=True)
    test_dataset = tonic.datasets.SHD(save_to=str(data_path), transform=transform, train=False)
    if cache:
        cache_suffix = "_binary" if one_spike_per_channel_per_timestep else ""
        train_dataset = DiskCachedDataset(
            train_dataset,
            transform=torch.from_numpy,
            cache_path=str(cache_path / f"SHD_t{timestep_size}_s{sensor_width}{cache_suffix}_train"),
        )
        test_dataset = DiskCachedDataset(
            test_dataset,
            transform=torch.from_numpy,
            cache_path=str(cache_path / f"SHD_t{timestep_size}_s{sensor_width}{cache_suffix}_test"),
        )
    input_shape = (sensor_width, 1, 1)
    n_outputs = SHD_N_OUTPUTS
    return train_dataset, test_dataset, input_shape, n_outputs, T
