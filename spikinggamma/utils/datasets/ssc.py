import torch
import tonic
from tonic import DiskCachedDataset, transforms

from spikinggamma.utils.datasets.common import (
    BinarizeSpikeCounts,
    TimeSliceTransform,
)

SSC_SAMPLE_LENGTH_US = 1000000
SSC_SOURCE_SENSOR_WIDTH = 700
SSC_N_OUTPUTS = 35


def get_ssc_dataset(
    data_path,
    cache_path,
    timestep_size=3600,
    sensor_width=140,
    cache=True,
    one_spike_per_channel_per_timestep=False,
):
    """Load SSC train/test splits with the same framing options as SHD."""
    data_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    T = SSC_SAMPLE_LENGTH_US // timestep_size
    frame_transforms = [
        transforms.Downsample(spatial_factor=sensor_width / SSC_SOURCE_SENSOR_WIDTH),
        transforms.ToFrame(sensor_size=(sensor_width, 1, 1), time_window=timestep_size),
        torch.from_numpy,
    ]
    if one_spike_per_channel_per_timestep:
        frame_transforms.append(BinarizeSpikeCounts())
    frame_transforms.append(TimeSliceTransform(time_steps=T))
    transform = transforms.Compose(frame_transforms)

    train_dataset = tonic.datasets.SSC(
        save_to=str(data_path), transform=transform, split="train",
    )
    test_dataset = tonic.datasets.SSC(
        save_to=str(data_path), transform=transform, split="test",
    )
    if cache:
        cache_suffix = "_binary" if one_spike_per_channel_per_timestep else ""
        train_dataset = DiskCachedDataset(
            train_dataset,
            transform=torch.from_numpy,
            cache_path=str(cache_path / f"SSC_t{timestep_size}_s{sensor_width}{cache_suffix}_train"),
        )
        test_dataset = DiskCachedDataset(
            test_dataset,
            transform=torch.from_numpy,
            cache_path=str(cache_path / f"SSC_t{timestep_size}_s{sensor_width}{cache_suffix}_test"),
        )
    input_shape = (sensor_width, 1, 1)
    return train_dataset, test_dataset, input_shape, SSC_N_OUTPUTS, T
