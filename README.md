# Introduction

This package allows you to directly use the SpikingGamma neuron model in your neural networks. It works on top of PyTorch.

# Installation

Install SpikingGamma from PyPI:

```console
pip install spikinggamma
```

PyTorch is installed automatically if it is missing; to use a specific CUDA build, install PyTorch first.

The dataset loaders in `spikinggamma.utils.datasets` and the example notebooks also need `tonic`, which the `datasets` extra installs:

```console
pip install "spikinggamma[datasets]"
```

For development, you can install directly from the source:

```console
git clone https://github.com/RoelMK/SpikingGamma
cd SpikingGamma
pip install -e .
```

To include the dataset dependencies in an editable install:

```console
pip install -e ".[datasets]"
```

# Tutorials

You can start by exploring the notebooks in the `examples` folder. Each example is a Jupyter notebook that can be run directly in a Jupyter environment.

| Notebook | Description |
|---|---|
| 1_getting_started.ipynb | Learn the basics: train and test a simple model. |
| 2_scaling_up.ipynb | Build and train a larger CNN. |
| 3_speeding_up.ipynb | Accelerate training through parallelization. |
