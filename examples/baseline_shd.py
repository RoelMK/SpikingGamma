# Run this Python script to reproduce the results for SHD from the paper.
# Should converge to about 92-93% test accuracy within 25 epochs.
# The model will be saved to "shd_model.pth".

import torch
import torch.nn as nn
import tonic
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from spikinggamma.neuron import SpikingGammaNeuron
from spikinggamma.signal.dropout import LockedDropout
from spikinggamma.spike.sigma_delta import SigmaDeltaSpikeFunction
from spikinggamma.dendrite.split import SplitDendriticFunction
from spikinggamma.dendrite.synapse.linear import LinearPerNeuron
from spikinggamma.signal.default import DefaultSignalFunction
from spikinggamma.spike.base import SpikeConfig
from spikinggamma.utils.enums import StepMode
from spikinggamma.utils.helpers import MultiStepCrossEntropyLoss
from spikinggamma.utils.datasets.shd import get_shd_dataset
from spikinggamma.spike.kernel.gamma import GammaKernelFunction, compute_linspace_bucket_decay_rates
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data"
cache_path = project_dir / "cache"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Data
train_dataset, test_dataset, input_shape, n_outputs, T = get_shd_dataset(data_path=data_path, cache_path=cache_path, timestep_size=3600)

# Hyperparameters
n_kernels = 10
decay_factor = 0.15
spike_config = SpikeConfig(
    base_spike_threshold=0.2,
    spike_grad_multiplier=1.0,
    use_dynamic_threshold=True,
    adaptation_mf=0.2
)
bucket_decay_rates = compute_linspace_bucket_decay_rates(n_kernels, decay_factor)

# A simple fully connected network with 3 hidden layers
class SpikingGammaNet(nn.Module):
    def __init__(self, 
                 input_shape, 
                 n_outputs,
                 n_kernels,
                 bucket_decay_rates,
                 spike_config):
        super().__init__()
        layer1_shape_out = (256,)
        layer2_shape_out = (256,)
        layer3_shape_out = (256,)
        layer4_shape_out = (n_outputs,)

        self.input_fn = SigmaDeltaSpikeFunction(GammaKernelFunction(bucket_decay_rates), spike_config)
        self.hid1 = SpikingGammaNeuron(
            dendritic_fn=SplitDendriticFunction(
                layer1_shape_out,
                n_kernels=n_kernels,
                synaptic_function=LinearPerNeuron(input_shape[0], layer1_shape_out[0])
            ),
            signal_fn=DefaultSignalFunction(
                pre_act_fn=nn.LayerNorm(layer1_shape_out),
                post_act_fn=LockedDropout(0.1),
            ),
            spike_fn=SigmaDeltaSpikeFunction(GammaKernelFunction(bucket_decay_rates), spike_config),
        )

        self.hid2 = SpikingGammaNeuron(
            dendritic_fn=SplitDendriticFunction(
                layer2_shape_out,
                n_kernels=n_kernels,
                synaptic_function=LinearPerNeuron(layer1_shape_out[0], layer2_shape_out[0])
            ),
            signal_fn=DefaultSignalFunction(
                pre_act_fn=nn.LayerNorm(layer2_shape_out),
                post_act_fn=LockedDropout(0.1),
            ),
            spike_fn=SigmaDeltaSpikeFunction(GammaKernelFunction(bucket_decay_rates), spike_config),
        )

        self.hid3 = SpikingGammaNeuron(
            dendritic_fn=SplitDendriticFunction(
                layer3_shape_out,
                n_kernels=n_kernels,
                synaptic_function=LinearPerNeuron(layer2_shape_out[0], layer3_shape_out[0])
            ),
            signal_fn=DefaultSignalFunction(
                pre_act_fn=nn.LayerNorm(layer3_shape_out),
                post_act_fn=LockedDropout(0.1),
            ),
            spike_fn=SigmaDeltaSpikeFunction(GammaKernelFunction(bucket_decay_rates), spike_config),
        )

        self.out = SpikingGammaNeuron(
            dendritic_fn=SplitDendriticFunction(
                layer4_shape_out,
                n_kernels=n_kernels,
                synaptic_function=LinearPerNeuron(layer3_shape_out[0], layer4_shape_out[0])
            ),
            signal_fn=DefaultSignalFunction(pre_act_fn=nn.LayerNorm(layer4_shape_out))
        )

    def reset(self):
        self.input_fn.reset()
        self.hid1.reset()
        self.hid2.reset()
        self.hid3.reset()
        self.out.reset()

    def detach(self):
        self.input_fn.detach()
        self.hid1.detach()
        self.hid2.detach()
        self.hid3.detach()
        self.out.detach()

    def forward(self, s_in, step_mode: StepMode):
        s_in = s_in.flatten(start_dim=1 if step_mode == StepMode.SINGLE else 2)
        s_in = s_in.to(dtype=self.input_fn.spike_threshold.dtype)
        y_hat_k_in, s_in = self.input_fn(s_in, step_mode=step_mode)
        y_hat_k_hid1, s_hid1 = self.hid1(y_hat_k_in, step_mode=step_mode)
        y_hat_k_hid2, s_hid2 = self.hid2(y_hat_k_hid1, step_mode=step_mode)
        y_hat_k_hid3, s_hid3 = self.hid3(y_hat_k_hid2, step_mode=step_mode)
        y_out = self.out(y_hat_k_hid3, step_mode=step_mode)
        # Train on the continuous output signal; evaluation sums it over time.
        return y_out
    

train_batch_size = 32
test_batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=False))
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=False))
model = SpikingGammaNet(input_shape, n_outputs, n_kernels, bucket_decay_rates, spike_config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = MultiStepCrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

step_mode = StepMode.MULTI_STATELESS
model.train()
n_epochs = 25

for _ in range(n_epochs):
    batch_iter = iter(train_dataloader)
    pbar = tqdm(batch_iter, total=len(train_dataloader))
    for data, labels in pbar:
        data = data.to(device)
        labels = labels.to(device)

        model.reset()
        optimizer.zero_grad()
        y = model(data, step_mode=step_mode)
        loss = loss_fn(y, labels)
        loss.backward()

        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():.4f}")
    lr_scheduler.step()

step_mode = StepMode.MULTI_STATELESS 
batch_iter = iter(test_dataloader)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    pbar = tqdm(batch_iter, total=len(test_dataloader))
    for data, labels in pbar:
        data = data.to(device)
        labels = labels.to(device)

        model.reset()
        y = model(data, step_mode=step_mode)
        predictions = y.sum(dim=0).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')

torch.save(model.state_dict(), "shd_model.pth")
