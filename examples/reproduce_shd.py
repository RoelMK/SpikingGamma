# Run this Python script to reproduce the results for SHD from the paper.
# Should converge to about 92-93% test accuracy within 25 epochs.
# The model will be saved to "shd_model.pth".

import torch
import torch.nn as nn
import tonic
from torch.utils.data import DataLoader
from tqdm import tqdm
from spikinggamma.neuron import SpikingGammaNeuron, SpikingGammaOutput
from spikinggamma.spike.sigma_delta import SigmaDeltaSpikeFunction
from spikinggamma.dendrite.split import SplitDendriticFunction
from spikinggamma.dendrite.synapse.linear import LinearPerNeuron
from spikinggamma.signal.default import DefaultSignalFunction
from spikinggamma.spike.base import compute_linspace_bucket_decay_rates, SpikeConfig
from spikinggamma.utils.enums import StepMode
from spikinggamma.utils.helpers import MultiStepCrossEntropyLoss
from load_data import get_shd_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Data
train_dataset, test_dataset, input_shape, n_outputs, T = get_shd_dataset(timestep_size=3600)

# Hyperparameters
n_buckets = 10
decay_factor = 0.15
spike_config = SpikeConfig(
    spike_threshold=0.2,
    spike_grad_multiplier=1.0
)
bucket_decay_rates = compute_linspace_bucket_decay_rates(n_buckets, decay_factor)

# A simple fully connected network with 3 hidden layers
class SpikingGammaNet(nn.Module):
    def __init__(self, 
                 input_shape, 
                 n_outputs,
                 n_buckets,
                 bucket_decay_rates,
                 spike_config):
        super().__init__()
        layer1_shape_out = (256,)
        layer2_shape_out = (256,)
        layer3_shape_out = (256,)
        layer4_shape_out = (n_outputs,)

        self.hid1 = SpikingGammaNeuron(
            spike_fn=SigmaDeltaSpikeFunction(bucket_decay_rates, spike_config),
            dendritic_fn=SplitDendriticFunction(
                layer1_shape_out,
                n_buckets=n_buckets,
                synaptic_function=LinearPerNeuron(input_shape[0], layer1_shape_out[0])
            ),
            signal_fn=DefaultSignalFunction(norm_fn=nn.LayerNorm(layer1_shape_out), dropout_fn=nn.Dropout(0.1))
        )

        self.hid2 = SpikingGammaNeuron(
            spike_fn=SigmaDeltaSpikeFunction(bucket_decay_rates, spike_config),
            dendritic_fn=SplitDendriticFunction(
                layer2_shape_out,
                n_buckets=n_buckets,
                synaptic_function=LinearPerNeuron(layer1_shape_out[0], layer2_shape_out[0])
            ),
            signal_fn=DefaultSignalFunction(norm_fn=nn.LayerNorm(layer2_shape_out), dropout_fn=nn.Dropout(0.1))
        )

        self.hid3 = SpikingGammaNeuron(
            spike_fn=SigmaDeltaSpikeFunction(bucket_decay_rates, spike_config),
            dendritic_fn=SplitDendriticFunction(
                layer3_shape_out,
                n_buckets=n_buckets,
                synaptic_function=LinearPerNeuron(layer2_shape_out[0], layer3_shape_out[0])
            ),
            signal_fn=DefaultSignalFunction(norm_fn=nn.LayerNorm(layer3_shape_out), dropout_fn=nn.Dropout(0.1))
        )

        self.out = SpikingGammaNeuron(
            spike_fn=SigmaDeltaSpikeFunction(bucket_decay_rates, spike_config),
            dendritic_fn=SplitDendriticFunction(
                layer4_shape_out,
                n_buckets=n_buckets,
                synaptic_function=LinearPerNeuron(layer3_shape_out[0], layer4_shape_out[0])
            ),
            signal_fn=DefaultSignalFunction(norm_fn=nn.LayerNorm(layer4_shape_out))
        )

        self.spike_out = SpikingGammaOutput(
            spike_fn=SigmaDeltaSpikeFunction(bucket_decay_rates, spike_config)
        )

    def reset(self):
        self.hid1.reset()
        self.hid2.reset()
        self.hid3.reset()
        self.out.reset()
        self.spike_out.reset()

    def detach(self):
        self.hid1.detach()
        self.hid2.detach()
        self.hid3.detach()
        self.out.detach()
        self.spike_out.detach()

    def forward(self, s_in, step_mode: StepMode):
        s_in = s_in.flatten(start_dim=1 if step_mode == StepMode.SINGLE else 2)
        y_hid1 = self.hid1(s_in, step_mode)
        y_hid2 = self.hid2(y_hid1, step_mode)
        y_hid3 = self.hid3(y_hid2, step_mode)
        y_out = self.out(y_hid3, step_mode)
        y_hat_out, s_out = self.spike_out(y_out, step_mode)
        return y_out, y_hat_out, s_out
    

train_batch_size = 32
test_batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=False))
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=False))
model = SpikingGammaNet(input_shape, n_outputs, n_buckets, bucket_decay_rates, spike_config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = MultiStepCrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
SigmaDeltaSpikeFunction.graph_mode = "1step"

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
        y, y_hat, s = model(data, step_mode=step_mode)
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
        y, y_hat, s = model(data, step_mode=step_mode)
        predictions = s.sum(dim=0).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')

torch.save(model.state_dict(), "shd_model.pth")