
# Introduction
This package allows you to directly use the SpikingGamma neuron model in your neural networks. It works on top of PyTorch. Check our paper for all details: https://openreview.net/pdf?id=ARDsBYnarO

# Installation
SpikingGamma is not yet on PyPI. At this moment, only installation from the source is supported.

To install directly from the source:
```console
pip install git+https://github.com/RoelMK/SpikingGamma
```
or
```console
git clone https://github.com/RoelMK/SpikingGamma
cd SpikingGamma
pip install -e .
```
PyTorch must be installed manually. For the example notebooks, make sure to install `tonic` as well.

# Tutorials
You can start by exploring the notebooks in the `examples` folder. Each example is a Jupyter notebook that can be run directly in a Jupyter environment.

| Notebook | Description |
|---|---|
| 1_getting_started.ipynb | Learn the basics: train and test a simple model. |
| 2_scaling_up.ipynb | Build and train a larger CNN. |
| 3_speeding_up.ipynb | Accelerate training through parallelization. |
