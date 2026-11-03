from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class BaseKernelState:
    def compute_y_hat_k(self) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def is_empty(self) -> bool:
        raise NotImplementedError

    def is_matching_shape(self, x: torch.Tensor) -> bool:
        raise NotImplementedError

    def initialize_like(self, x: torch.Tensor, n_kernels: int):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def detach(self):
        raise NotImplementedError

    def init_from(self, other: "BaseKernelState"):
        raise NotImplementedError


@dataclass
class YHatKernelState(BaseKernelState):
    # A plain field: torch.compile (Dynamo) misses writes made through a property setter
    # when methods read the field behind it, which made compiled encoders return zeros.
    y_hat_k: Optional[torch.Tensor] = None

    def compute_y_hat_k(self) -> Optional[torch.Tensor]:
        if self.y_hat_k is None:
            return None
        return self.y_hat_k.clone()

    def is_empty(self) -> bool:
        return self.y_hat_k is None

    def is_matching_shape(self, x: torch.Tensor) -> bool:
        return self.y_hat_k is not None and self.y_hat_k.shape[:-1] == x.shape

    def initialize_like(self, x: torch.Tensor, n_kernels: int):
        self.y_hat_k = torch.zeros((*x.shape, n_kernels), device=x.device, dtype=x.dtype)

    def reset(self):
        self.y_hat_k = None

    def detach(self):
        if self.y_hat_k is not None:
            self.y_hat_k = self.y_hat_k.detach()

    def init_from(self, other: "YHatKernelState"):
        if not isinstance(other, YHatKernelState):
            raise TypeError("init_from expects an instance of YHatKernelState")
        self.y_hat_k = other.y_hat_k.clone() if other.y_hat_k is not None else None


def prototype_y_hat_k(y_hat_k: Optional[torch.Tensor], prototype_shape: torch.Size, n_kernels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if y_hat_k is not None:
        return y_hat_k
    else:
        return torch.zeros((*prototype_shape, n_kernels), device=device, dtype=dtype)


class BaseKernelFunction:
    n_kernels: int

    def __init__(self, n_kernels: int):
        super().__init__()
        self.n_kernels = n_kernels

    def construct_kernel_state(self) -> BaseKernelState:
        raise NotImplementedError
    
    @property
    def normal_saturation_cutoff(self) -> torch.Tensor:
        raise NotImplementedError
    
    @torch.no_grad()
    def step(
        self,
        state: BaseKernelState,
    ):
        raise NotImplementedError
        
    def add(self, state: BaseKernelState, x: torch.Tensor):
        raise NotImplementedError
        