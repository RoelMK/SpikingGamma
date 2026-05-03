# Deep CNN Example

This example trains a deeper VGG-style convolutional SpikingGamma network on
DVS Gesture. It is intended as a compact scaling example: the same components
used in smaller experiments are applied in a deeper CNN setting, with run
configuration, checkpoints, and metrics written under `runs/dvsgesture`.

Basic usage from this directory:

```bash
python run_dvsgesture.py
```

The script uses sensible defaults and exposes the main training, model, data,
and spike parameters as command-line arguments for adjustments.
