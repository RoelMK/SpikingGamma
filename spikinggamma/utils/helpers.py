from typing import Optional
import torch
import torch.nn as nn

from spikinggamma.utils.enums import StepMode

class MultiStepLayerWrapper(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x, step_mode: StepMode = StepMode.MULTI_STATELESS):
        if step_mode == StepMode.MULTI_STATEFUL or step_mode == StepMode.MULTI_STATELESS:
            T, B, *S = x.shape
            x = x.view(T * B, *S)  # Combine time and batch dimensions
        x = self.layer(x)
        if step_mode == StepMode.MULTI_STATEFUL or step_mode == StepMode.MULTI_STATELESS:
            x = x.view(T, B, *x.shape[1:])  # Reshape back to (T, B, ...)
        return x
    

class MultiStepCrossEntropyLoss(nn.Module):
    """
    Args:
      - y: (T, B, C) logits
      - target: (B,) integer class labels
    Optional:
      - mask: (T, B) boolean to ignore padded time steps
    """
    def __init__(self):
        super().__init__()
        self.inner_loss = nn.CrossEntropyLoss()

    def forward(self, y, target, mask: Optional[torch.Tensor] = None):
        T, B, C = y.shape
        y_flat = y.reshape(T * B, C)
        tgt_TB = target[None, :].expand(T, B).reshape(T * B)

        if mask is not None:
            mask_flat = mask.reshape(T * B)
            y_flat = y_flat[mask_flat]
            tgt_TB = tgt_TB[mask_flat]

        return self.inner_loss(y_flat, tgt_TB)
    

def multi_split_step_forward(data: torch.Tensor, labels: torch.Tensor, model: nn.Module, loss_fn: nn.Module, T_per_split: int, 
                             detach_y = True, detach_y_hat = True, detach_s = True):
    """
    Perform a forward and backward pass in multiple splits along the time dimension.
    This is useful when the full sequence does not fit in memory.
    Args:
      - data: (T, B, ...) input data
      - labels: what is passed to loss_fn
      - model: the model
      - loss_fn: loss function, should return a scalar
      - T_per_split: number of timesteps per split, must be in [1, T] and ideally divide T evenly
      - detach_y, detach_y_hat, detach_s: whether to detach these outputs from the computation graph to save memory
    Returns:
      - y: (T, B, ...) neuron signal
      - y_hat: (T, B, ..., K) bucket outputs, for K buckets
      - s: (T, B, ...) spike outputs
      - total_loss: scalar loss value
    """

    # Assumes data is (T, B, *S)
    if T_per_split < 1 or T_per_split > data.shape[0]:
        raise ValueError("T_per_split must be in [1, T]")
    if data.shape[0] % T_per_split != 0:
        print(f"Warning: T_per_split {T_per_split} does not evenly divide T {data.shape[0]}. The last split will be smaller. This may affect performance when using CUDA graphs.")
    
    y_list, y_hat_list, s_list = [], [], []
    total_loss = 0
    for t_start in range(0, data.shape[0], T_per_split):
        t_end = min(t_start + T_per_split, data.shape[0])
        data_split = data[t_start:t_end]
        y, y_hat, s = model(data_split, step_mode=StepMode.MULTI_STATEFUL)
        loss = loss_fn(y, labels) * (data_split.shape[0] / data.shape[0])
        loss.backward()
        y_list.append(y.detach() if detach_y else y)
        y_hat_list.append(y_hat.detach() if detach_y_hat else y_hat)
        s_list.append(s.detach() if detach_s else s)
        total_loss += loss.item()

    y = torch.cat(y_list, dim=0)
    y_hat = torch.cat(y_hat_list, dim=0)
    s = torch.cat(s_list, dim=0)
    return y, y_hat, s, total_loss