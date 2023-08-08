# Trainable Fractional Fourier Transform

[![tox](https://github.com/tunakasif/torch-frft/actions/workflows/tox.yml/badge.svg?branch=main)](https://github.com/tunakasif/torch-frft/actions/workflows/tox.yml)
[![commitizen](https://github.com/tunakasif/torch-frft/actions/workflows/bump.yml/badge.svg?branch=main)](https://github.com/tunakasif/torch-frft/actions/workflows/bump.yml)
[![Python Versions](https://img.shields.io/badge/python-3.10%20|%203.11-blue.svg)](https://img.shields.io/badge/python-3.10%20|%203.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)

A differentiable fractional Fourier transform (FRFT) implementation with layers that can be trained end-to-end with the rest of the network. This package provides implementations of both fast computation of continuous FRFT and discrete FRFT (DFRFT) and also provides pre-configured layers that can be used in neural networks.

The fast transform is an approximation of the continuous FRFT and is based on [_Digital computation of the fractional Fourier transform_](https://ieeexplore.ieee.org/document/536672) paper. The DFRFT is based on [_The discrete fractional Fourier transform_](https://ieeexplore.ieee.org/document/839980) paper. MATLAB implementations of both approaches are provided on [Haldun M. Özaktaş's page](http://www.ee.bilkent.edu.tr/~haldun/wileybook.html) as [`fracF.m`](http://www.ee.bilkent.edu.tr/~haldun/fracF.m) and [`dFRT.m`](http://www.ee.bilkent.edu.tr/~haldun/dFRT.m), respectively.

This package implements these approaches in PyTorch with certain optimazations and most importantly adds the ability to apply the transform along a certain dimension of a tensor.

With the implemented transforms basic layers that extend `torch.nn.Module` is provided, an example of the custom layer implementation is also provided in [`README.md`](#custom-layers) file.

## Table of Contents

- [Trainable Fractional Fourier Transform](#trainable-fractional-fourier-transform)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [For Usage](#for-usage)
      - [Using `pip`](#using-pip)
      - [Using `poetry`](#using-poetry)
    - [For Development](#for-development)
  - [Usage](#usage)
    - [Transforms](#transforms)
    - [Pre-configured Layers](#pre-configured-layers)
    - [Custom Layers](#custom-layers)

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

:warning: Transforms applied in the same device as the input tensor. If the input tensor is on GPU, the transform will be applied on GPU as well.

The package provides transform functions that operate on the $n^{th}$ dimension of an input tensor, `frft()` and `dfrft()`, which correspond to the fast computation of continuous fractional Fourier transform (FRFT) and discrete fractional Fourier transform (DFRFT), respectively. It also provides a function `dfrftmtx()` which computes the DFRFT matrix for a given length and order similar to MATLAB's `dftmtx()` function for ordinary DFT matrix. Note that the `frft()` only operates on even-sized lengths as in the original MATLAB implementation [fracF.m](http://www.ee.bilkent.edu.tr/~haldun/fracF.m).

```python
import torch
from torch_frft.frft_module import frft
from torch_frft.dfrft_module import dfrft, dfrftmtx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 128
a = 0.5
X = torch.rand(N, N, device=device)
Y1 = frft(X, a)  # equivalent to dim=-1
Y2 = frft(X, a, dim=0)

# 2D FRFT
a0, a1 = 1.25, 0.75
Y3 = frft(frft(X, a0, dim=0), a1, dim=1)
```

### Pre-configured Layers

The package also provides two differentiable FRFT layers, `FrFTLayer` and `DFrFTLayer`, which can be used as follows:

```python
import torch
import torch.nn as nn

from torch_frft.dfrft_module import dfrft
from torch_frft.layer import DFrFTLayer, FrFTLayer

# FRFT with initial order 1.25, operating on the last dimension
model = nn.Sequential(FrFTLayer(order=1.25, dim=-1))

# DFRFT with initial order 0.75, operating on the first dimension
model = nn.Sequential(DFrFTLayer(order=0.75, dim=0))
```

Then, the simplest toy example to train the layer is as follows:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples, seq_length = 100, 16
a_original = 1.1
a_initial = 1.25
X = torch.randn(num_samples, seq_length, dtype=torch.float32, device=device)
Y = dfrft(X, a_original)

model = DFrFTLayer(order=a_initial).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 1000

for epoch in range(1 + epochs):
    optim.zero_grad()
    loss = torch.norm(Y - model(X))
    loss.backward()
    optim.step()

print("Original  a:", a_original)
print("Estimated a:", model.order.item())
```

Note that you can also place these layers directly into `torch.nn.Sequential`. Remark that these transforms generate complex-valued outputs, so you may need to convert them to real-valued outputs, e.g., taking the real part, absolute value, etc. For example, the following code snippet implements a simple fully-connected network with real parts of `FrFTLayer` and `DFrFTLayer` in between:

```python
class Real(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.real


model = nn.Sequential(
    nn.Linear(16, 6),
    nn.ReLU(),
    DFrFTLayer(1.35, dim=-1),
    Real(),
    nn.ReLU(),
    nn.Linear(6, 1),
    FrFTLayer(0.65, dim=0),
    Real(),
    nn.ReLU(),
)
```

### Custom Layers

It is also possible to create custom layers with the provided `frft()` and `dfrft()` functions. The below example contains in-between `Linear` and `ReLU` layers and the same fractional order for forward and backward DFRFT transforms is as follows:

```python
import torch
import torch.nn as nn
from torch_frft.dfrft_module import dfrft


class CustomLayer(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, *, order: float = 1.0, dim: int = -1) -> None:
        super().__init__()
        self.in_features = in_feat
        self.out_features = out_feat
        self.order = nn.Parameter(torch.tensor(order, dtype=torch.float32), requires_grad=True)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = dfrft(x, self.order, dim=self.dim)
        a1 = nn.ReLU()(x1.real) + 1j * nn.ReLU()(x1.imag)
        x2 = nn.Linear(self.in_features, self.in_features, dtype=a1.dtype, device=x.device)(a1)
        a2 = nn.ReLU()(x2.real) + 1j * nn.ReLU()(x2.imag)
        x3 = dfrft(a2, -self.order, dim=self.dim)
        a3 = nn.ReLU()(x3.real) + 1j * nn.ReLU()(x3.imag)
        x4 = nn.Linear(self.in_features, self.out_features, dtype=a3.dtype, device=x.device)(a3)
        a4 = nn.ReLU()(x4.real)
        return a4
```

Then, a simple training example for the given `CustomLayer` can be given as follows:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples, seq_length, out_length = 100, 32, 5
X = torch.rand(num_samples, seq_length, device=device)
y = torch.rand(num_samples, out_length, device=device)

model = CustomLayer(seq_length, out_length, order=1.25)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 1000

for epoch in range(epochs):
    optim.zero_grad()
    loss = torch.nn.MSELoss()
    output = loss(model(X), y)
    output.backward()
    optim.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss {output.item():.4f}")
print("Final a:", model.order)
```
