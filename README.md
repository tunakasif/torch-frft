# Trainable Fractional Fourier Transform

[![tox](https://github.com/tunakasif/torch-frft/actions/workflows/tox.yml/badge.svg?branch=main)](https://github.com/tunakasif/torch-frft/actions/workflows/tox.yml)
[![commitizen](https://github.com/tunakasif/torch-frft/actions/workflows/bump.yml/badge.svg?branch=main)](https://github.com/tunakasif/torch-frft/actions/workflows/bump.yml)
[![Python Versions](https://img.shields.io/badge/python-3.10%20|%203.11-blue.svg)](https://img.shields.io/badge/python-3.10%20|%203.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)

A differentiable fractional Fourier transform (FRFT) layer, which can be trained end-to-end with the rest of the network.

## Installation

### For Usage

This GitHub repository is `pip`/`poetry` installable. You can install the package using `pip` or `poetry` as follows:

#### Using `pip`

```sh
pip install git+https://github.com/tunakasif/torch-frft.git
```

#### Using `poetry`

Either run the following command:

```sh
poetry add git+https://github.com/tunakasif/torch-frft.git
```

or add the following to your `pyproject.toml` file:

```toml
[tool.poetry.dependencies]
torch-frft = {git = "github.com/tunakasif/torch-frft.git"}
```

For simplicity `https` version is provided, if you prefer you may also use the `ssh` version as well.

### For Development

This codebase utilizes [`Poetry`](https://python-poetry.org) for package management. To install the dependencies, run:

```sh
poetry install
```

or if you do not want to use `Poetry`, you can install the dependencies provided in [`requirements.txt`](requirements.txt) using `pip` or `conda`, e,g.,

```sh
pip install -r requirements.txt
```

## Usage

### Transforms

The package provides transform functions that operate on the $n^{th}$ dimension of an input tensor, `frft()` and `dfrft()`, which correspond to the fast computation of continuous fractional Fourier transform (FRFT) and discrete fractional Fourier transform (DFRFT), respectively. It also provides a function `dfrftmtx()` which computes the DFRFT matrix for a given length and order similar to MATLAB's `dftmtx()` function for ordinary DFT matrix.

```python
import torch
from torch_frft.frft_module import frft
from torch_frft.dfrft_module import dfrft, dfrftmtx

N = 128
a = 0.5
X = torch.rand(N, N)
Y1 = frft(X, a) # equivalent to dim=-1
Y2 = frft(X, a, dim=0)

# 2D FRFT
a0 = 1.25
a1 = 0.75
Y3 = frft(frft(X, a0, dim=0), a1, dim=1)
```

### Layers

The package also provides two differentiable FRFT layers, `FrFTLayer` and `DFrFTLayer`, which can be used as follows:

```python
import torch
import torch.nn as nn
from torch_frft.layer import DFrFTLayer, FrFTLayer

# FRFT with initial order 1.25, operating on the last dimension
model = nn.Sequential(FrFTLayer(order=1.25, dim=-1))

# DFRFT with initial order 0.75, operating on the first dimension
model = nn.Sequential(DFrFTLayer(order=0.75, dim=0))
```

Then, the simplest toy example to train the layer is as follows:

```python
num_samples, seq_length = 100, 16
a_original = torch.tensor(1.1, dtype=torch.float32)
X = torch.randn(num_samples, seq_length, dtype=torch.float32)
Y = dfrft(X, a_original)

model = nn.Sequential(DFrFTLayer(order=1.25))
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 1000

for epoch in range(1 + epochs):
    optim.zero_grad()
    loss = torch.norm(Y - model(X))
    loss.backward()
    optim.step()

print("Original  a:", a_original)
print("Estimated a:", model[0].order)
```
