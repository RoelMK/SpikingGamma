import torch


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


class BinarizeSpikeCounts:
    def __call__(self, x: torch.Tensor):
        return (x > 0).to(x.dtype)
