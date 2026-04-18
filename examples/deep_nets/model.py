from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from spikinggamma.dendrite.split import SplitDendriticFunction
from spikinggamma.dendrite.synapse.conv2d import Conv2dPerNeuron, WSConv2dPerNeuron
from spikinggamma.dendrite.synapse.linear import LinearPerNeuron, WSLinearPerNeuron
from spikinggamma.neuron import SpikingGammaNeuron, SpikingGammaOutput
from spikinggamma.signal.default import DefaultSignalFunction
from spikinggamma.spike.base import SpikeConfig
from spikinggamma.spike.sigma_delta import SigmaDeltaSpikeFunction
from spikinggamma.utils.enums import StepMode


VGG_CONFIGS = {
    "vgg6": [
        (64, 1, "max"),
        (128, 2, "max"),
        (256, 1, "max"),
        (512, 2, "max"),
        (512, 2, "aavg"),
    ],
    "vgg9": [
        (64, 1, "max"),
        (128, 2, "max"),
        (256, 1, "max"),
        (256, 2, "max"),
        (512, 1, "max"),
        (512, 2, "max"),
        (512, 1, "max"),
        (512, 2, "aavg"),
    ],
    "vgg12": [
        (64, 1, "max"),
        (64, 2, "max"),
        (128, 1, "max"),
        (128, 2, "max"),
        (256, 1, "max"),
        (256, 1, "max"),
        (256, 2, "max"),
        (512, 1, "max"),
        (512, 1, "max"),
        (512, 1, "max"),
        (512, 2, "aavg"),
    ],
    "vgg15": [
        (64, 1, "max"),
        (64, 1, "max"),
        (128, 2, "max"),
        (128, 1, "max"),
        (256, 1, "max"),
        (256, 1, "max"),
        (256, 2, "max"),
        (512, 1, "max"),
        (512, 1, "max"),
        (512, 2, "max"),
        (512, 1, "max"),
        (512, 1, "max"),
        (512, 2, "max"),
        (512, 2, "aavg"),
    ],
}


@dataclass(frozen=True)
class LayerSpec:
    kind: str
    name: str


FEATURE_NORMS = {"layernorm", "ws", "none"}
CLASSIFIER_NORMS = {"layernorm", "ws", "none"}


def _single_timescale_decay_rates(bucket_decay_rates: torch.Tensor) -> torch.Tensor:
    if bucket_decay_rates.ndim == 1:
        return bucket_decay_rates[:1].clone()
    return bucket_decay_rates[..., :1].clone()


def _conv_output_dim(size: int, kernel_size: int, stride: int, padding: int, dilation: int = 1) -> int:
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def _pool_output_dim(size: int, kernel_size: int, stride: int, padding: int = 0, dilation: int = 1) -> int:
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def _adaptive_pool_output_dim(target_size: int) -> int:
    return target_size


class VGGSpikingGammaNet(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        n_outputs: int,
        classifier_bucket_decay_rates: torch.Tensor,
        spike_config: SpikeConfig,
        feature_cfg: Sequence[tuple[int, int, str]] = VGG_CONFIGS["vgg9"],
        classifier_hidden_dims: Sequence[int] = (),
        classifier_dropout: float = 0.0,
        feature_bucket_decay_rates: torch.Tensor | None = None,
        feature_norm: str = "ws",
        classifier_norm: str = "layernorm",
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"Expected input_shape to be (C, H, W), got {tuple(input_shape)}")
        if feature_norm not in FEATURE_NORMS:
            raise ValueError(f"Unknown feature_norm '{feature_norm}'. Available: {sorted(FEATURE_NORMS)}")
        if classifier_norm not in CLASSIFIER_NORMS:
            raise ValueError(f"Unknown classifier_norm '{classifier_norm}'. Available: {sorted(CLASSIFIER_NORMS)}")

        self.input_shape = tuple(int(v) for v in input_shape)
        self.n_outputs = int(n_outputs)
        self.feature_cfg = list(feature_cfg)
        self.classifier_hidden_dims = tuple(int(v) for v in classifier_hidden_dims)
        self.feature_norm = feature_norm
        self.classifier_norm = classifier_norm

        if feature_bucket_decay_rates is None:
            feature_bucket_decay_rates = _single_timescale_decay_rates(classifier_bucket_decay_rates)

        self.feature_layers, self.feature_specs, feature_shape = self._make_feature_extractor(
            feature_cfg=self.feature_cfg,
            input_shape=self.input_shape,
            bucket_decay_rates=feature_bucket_decay_rates,
            spike_config=spike_config,
            feature_norm=feature_norm,
        )
        self.feature_shape = feature_shape
        self.feature_dim = feature_shape[0] * feature_shape[1] * feature_shape[2]

        self.classifier_layers, self.classifier_specs = self._make_classifier(
            input_dim=self.feature_dim,
            hidden_dims=self.classifier_hidden_dims,
            output_dim=self.n_outputs,
            bucket_decay_rates=classifier_bucket_decay_rates,
            spike_config=spike_config,
            dropout=classifier_dropout,
            classifier_norm=classifier_norm,
        )

        self.spike_out = SpikingGammaOutput(
            spike_fn=SigmaDeltaSpikeFunction(classifier_bucket_decay_rates, spike_config)
        )

    def _make_feature_extractor(
        self,
        feature_cfg: Sequence[tuple[int, int, str]],
        input_shape: Sequence[int],
        bucket_decay_rates: torch.Tensor,
        spike_config: SpikeConfig,
        feature_norm: str,
    ) -> tuple[nn.ModuleList, list[LayerSpec], tuple[int, int, int]]:
        channels, height, width = input_shape
        layers = nn.ModuleList()
        specs: list[LayerSpec] = []
        conv_idx = 0
        pool_idx = 0

        for out_channels, block_pool_size, block_pool_type in feature_cfg:
            conv_idx += 1
            shape_out = (
                int(out_channels),
                _conv_output_dim(height, kernel_size=3, stride=1, padding=1),
                _conv_output_dim(width, kernel_size=3, stride=1, padding=1),
            )
            layers.append(
                SpikingGammaNeuron(
                    spike_fn=SigmaDeltaSpikeFunction(bucket_decay_rates, spike_config),
                    dendritic_fn=SplitDendriticFunction(
                        shape_out=shape_out,
                        n_buckets=bucket_decay_rates.shape[-1],
                        synaptic_function=self._build_conv_synapse(
                            feature_norm=feature_norm,
                            in_channels=channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    signal_fn=DefaultSignalFunction(
                        norm_fn=self._build_feature_norm(feature_norm=feature_norm, shape_out=shape_out),
                    ),
                )
            )
            specs.append(LayerSpec(kind="conv", name=f"conv{conv_idx}"))
            channels, height, width = shape_out

            pool_layer, height, width = self._build_pool_layer(
                pool_size=block_pool_size,
                pool_type=block_pool_type,
                height=height,
                width=width,
            )
            if pool_layer is not None:
                pool_idx += 1
                layers.append(pool_layer)
                specs.append(LayerSpec(kind="pool", name=f"pool{pool_idx}"))

        return layers, specs, (channels, height, width)

    def _make_classifier(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        bucket_decay_rates: torch.Tensor,
        spike_config: SpikeConfig,
        dropout: float,
        classifier_norm: str,
    ) -> tuple[nn.ModuleList, list[LayerSpec]]:
        dims = [input_dim, *hidden_dims, output_dim]
        layers = nn.ModuleList()
        specs: list[LayerSpec] = []

        for idx, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:]), start=1):
            is_output_layer = idx == len(dims) - 1
            dropout_fn = nn.Dropout(dropout) if dropout > 0 and not is_output_layer else None
            norm_fn = self._build_classifier_norm(classifier_norm=classifier_norm, dim_out=dim_out) if not is_output_layer else None
            layers.append(
                SpikingGammaNeuron(
                    spike_fn=SigmaDeltaSpikeFunction(bucket_decay_rates, spike_config),
                    dendritic_fn=SplitDendriticFunction(
                        shape_out=(dim_out,),
                        n_buckets=bucket_decay_rates.shape[-1],
                        synaptic_function=self._build_classifier_synapse(
                            classifier_norm=classifier_norm,
                            dim_in=dim_in,
                            dim_out=dim_out,
                        ),
                    ),
                    signal_fn=DefaultSignalFunction(norm_fn=norm_fn, dropout_fn=dropout_fn),
                )
            )
            specs.append(LayerSpec(kind="classifier", name=f"fc{idx}"))

        return layers, specs

    def _build_conv_synapse(
        self,
        feature_norm: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool,
    ):
        conv_cls = WSConv2dPerNeuron if feature_norm == "ws" else Conv2dPerNeuron
        return conv_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def _build_feature_norm(self, feature_norm: str, shape_out: tuple[int, int, int]):
        if feature_norm == "layernorm":
            return nn.LayerNorm(shape_out)
        return None

    def _build_classifier_synapse(self, classifier_norm: str, dim_in: int, dim_out: int):
        if classifier_norm == "ws":
            return WSLinearPerNeuron(dim_in, dim_out, bias=True)
        return LinearPerNeuron(dim_in, dim_out)

    def _build_classifier_norm(self, classifier_norm: str, dim_out: int):
        if classifier_norm == "layernorm":
            return nn.LayerNorm((dim_out,))
        return None

    def _build_pool_layer(
        self,
        pool_size: int,
        pool_type: str,
        height: int,
        width: int,
    ):
        if pool_type == "max":
            if pool_size <= 1:
                return None, height, width
            kernel_size = pool_size
            stride = pool_size
            next_height = _pool_output_dim(height, kernel_size, stride)
            next_width = _pool_output_dim(width, kernel_size, stride)
            return nn.MaxPool2d(kernel_size=kernel_size, stride=stride), next_height, next_width

        if pool_type == "aavg":
            next_height = _adaptive_pool_output_dim(pool_size)
            next_width = _adaptive_pool_output_dim(pool_size)
            return nn.AdaptiveAvgPool2d((pool_size, pool_size)), next_height, next_width

        raise ValueError(f"Unknown pool_type '{pool_type}'")

    def reset(self):
        for layer, spec in zip(self.feature_layers, self.feature_specs):
            if spec.kind == "conv":
                layer.reset()
        for layer in self.classifier_layers:
            layer.reset()
        self.spike_out.reset()

    def detach(self):
        for layer, spec in zip(self.feature_layers, self.feature_specs):
            if spec.kind == "conv":
                layer.detach()
        for layer in self.classifier_layers:
            layer.detach()
        self.spike_out.detach()

    def _flatten_features(self, x: torch.Tensor, step_mode: StepMode) -> torch.Tensor:
        start_dim = 1 if step_mode == StepMode.SINGLE else 2
        return x.flatten(start_dim=start_dim)

    def forward(self, s_in: torch.Tensor, step_mode: StepMode = StepMode.SINGLE):
        y = s_in
        for layer, spec in zip(self.feature_layers, self.feature_specs):
            if spec.kind == "conv":
                y = layer(y, step_mode=step_mode)
            else:
                if step_mode == StepMode.SINGLE:
                    y = layer(y)
                else:
                    time_steps, batch_size = y.shape[:2]
                    y = y.reshape(time_steps * batch_size, *y.shape[2:])
                    y = layer(y)
                    y = y.reshape(time_steps, batch_size, *y.shape[1:])

        y = self._flatten_features(y, step_mode)

        for layer in self.classifier_layers:
            y = layer(y, step_mode=step_mode)

        y_hat_out, s_out = self.spike_out(y, step_mode=step_mode)
        return y, y_hat_out, s_out


def build_vgg_spiking_gamma_net(
    input_shape: Sequence[int],
    n_outputs: int,
    classifier_bucket_decay_rates: torch.Tensor,
    spike_config: SpikeConfig,
    architecture: str = "vgg9",
    classifier_hidden_dims: Iterable[int] = (),
    classifier_dropout: float = 0.0,
    feature_bucket_decay_rates: torch.Tensor | None = None,
    feature_norm: str = "ws",
    classifier_norm: str = "layernorm",
    **kwargs,
) -> VGGSpikingGammaNet:
    if architecture not in VGG_CONFIGS:
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {sorted(VGG_CONFIGS)}")

    return VGGSpikingGammaNet(
        input_shape=input_shape,
        n_outputs=n_outputs,
        classifier_bucket_decay_rates=classifier_bucket_decay_rates,
        spike_config=spike_config,
        feature_cfg=VGG_CONFIGS[architecture],
        classifier_hidden_dims=tuple(classifier_hidden_dims),
        classifier_dropout=classifier_dropout,
        feature_bucket_decay_rates=feature_bucket_decay_rates,
        feature_norm=feature_norm,
        classifier_norm=classifier_norm,
        **kwargs,
    )
