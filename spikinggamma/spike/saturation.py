from typing import Optional, Union
import torch

@torch.no_grad()
def compute_saturation_gate(
    y: torch.Tensor,
    cutoff: Union[float, torch.Tensor],
    softness: float,
    start_fraction: float,
) -> torch.Tensor:
    if torch.is_tensor(cutoff) and len(cutoff.shape) > 0:
        if cutoff.shape == y.shape[1:]:
            cutoff = cutoff.reshape((1, *cutoff.shape))
        elif y.ndim >= 3 and cutoff.shape == y.shape[2:]:
            cutoff = cutoff.reshape((1, 1, *cutoff.shape))
        else:
            expected_shapes = [tuple(y.shape[1:])]
            if y.ndim >= 3:
                expected_shapes.append(tuple(y.shape[2:]))
            raise ValueError(
                f"cutoff shape {cutoff.shape} does not match y feature shapes "
                f"{expected_shapes}"
            )

    if softness <= 0:
        gate = (y < cutoff).to(y.dtype)
    else:
        start = start_fraction * cutoff
        transition_width = softness * (-start + cutoff) + 1e-8

        excess = (y - start) / transition_width
        gate = torch.where(y <= start, torch.ones_like(y), torch.exp(-excess))

    if gate.shape != y.shape:
        gate = gate.expand_as(y)
    return gate

@torch.no_grad()
def apply_saturation_cutoff(dL_dy: torch.Tensor, y: Optional[torch.Tensor], saturation_cutoff: Optional[Union[float, torch.Tensor]], 
                            saturation_gate_softness: float, saturation_gate_start_fraction: float, can_saturate: bool) -> torch.Tensor:
    if not can_saturate:
        return dL_dy
    elif y is None or saturation_cutoff is None:
        raise ValueError("y and saturation_cutoff must be provided if can_saturate is True")
    gate = compute_saturation_gate(
        y,
        cutoff=saturation_cutoff,
        softness=saturation_gate_softness,
        start_fraction=saturation_gate_start_fraction,
    )
    return dL_dy * gate

@torch.no_grad()
def y_hat_backward_saturated(dL_dy_hat_k: torch.Tensor, y: Optional[torch.Tensor], saturation_cutoff: Optional[Union[float, torch.Tensor]], n_kernels: int,
                   saturation_gate_softness: float, saturation_gate_start_fraction: float, can_saturate: bool, spike_grad_multiplier: Optional[float] = None) -> torch.Tensor:
    if spike_grad_multiplier is None:
        spike_grad_multiplier = 1.0/n_kernels
    dL_dy = dL_dy_hat_k.sum(dim=-1) * spike_grad_multiplier
    dL_dy_saturated = apply_saturation_cutoff(dL_dy, y, saturation_cutoff, saturation_gate_softness, saturation_gate_start_fraction, can_saturate)
    return dL_dy_saturated