# Trainable Fractional Fourier Transform

[![PyPI](https://img.shields.io/pypi/v/torch-frft)](https://pypi.org/project/torch-frft)
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/torch-frft)](https://anaconda.org/conda-forge/torch-frft)
[![Tox & Bump Version](https://github.com/tunakasif/torch-frft/actions/workflows/build.yml/badge.svg)](https://github.com/tunakasif/torch-frft/actions/workflows/build.yml)
[![Codecov](https://img.shields.io/codecov/c/github/tunakasif/torch-frft)](https://app.codecov.io/gh/tunakasif/torch-frft)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch-frft)](https://pypi.org/project/torch-frft)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/torch-frft)](https://pypi.org/project/torch-frft)
[![GitHub](https://img.shields.io/github/license/tunakasif/torch-frft)](https://github.com/tunakasif/torch-frft/blob/main/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

A differentiable fractional Fourier transform (FRFT) implementation with layers that can be trained end-to-end with the rest of the network. This package provides implementations of both fast computations of continuous FRFT and discrete FRFT (DFRFT) and pre-configured layers that are eligible for use in neural networks.

The fast transform approximates the continuous FRFT and is based on [_Digital computation of the fractional Fourier transform_](https://ieeexplore.ieee.org/document/536672) paper. The DFRFT is based on [_The discrete fractional Fourier transform_](https://ieeexplore.ieee.org/document/839980) paper. MATLAB implementations of both approaches are provided on [Haldun M. Özaktaş's page](http://www.ee.bilkent.edu.tr/~haldun/wileybook.html) as [`fracF.m`](http://www.ee.bilkent.edu.tr/~haldun/fracF.m) and [`dFRT.m`](http://www.ee.bilkent.edu.tr/~haldun/dFRT.m), respectively.

This package implements these approaches in PyTorch with specific optimizations and, most notably, adds the ability to apply the transform along a particular tensor dimension.

We provide primer layers that extend `torch.nn.Module` for continuous and discrete transforms, an example of the custom layer implementation, is also provided in the `README.md` file.

We developed this project for the [_Trainable Fractional Fourier Transform_](https://ieeexplore.ieee.org/document/10458263) paper, published in _IEEE Signal Processing Letters_. You can also access the [paper's GitHub page](https://github.com/koc-lab/TrainableFrFT) for experiments and example usage. If you find this package useful, please consider citing as follows:

```bibtex
@article{trainable-frft-2024,
  author   = {Koç, Emirhan and Alikaşifoğlu, Tuna and Aras, Arda Can and Koç, Aykut},
  journal  = {IEEE Signal Processing Letters},
  title    = {Trainable Fractional Fourier Transform},
  year     = {2024},
  volume   = {31},
  number   = {},
  pages    = {751-755},
  keywords = {Vectors;Convolution;Training;Task analysis;Computational modeling;Time series analysis;Feature extraction;Machine learning;neural networks;FT;fractional FT;deep learning},
  doi      = {10.1109/LSP.2024.3372779}
}
```

## Table of Contents

- [Trainable Fractional Fourier Transform](#trainable-fractional-fourier-transform)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [For Usage](#for-usage)
    - [For Development](#for-development)
  - [Usage](#usage)
    - [Transforms](#transforms)
    - [Pre-configured Layers](#pre-configured-layers)
    - [Custom Layers](#custom-layers)
  - [FRFT Shift](#frft-shift)

## Installation

### For Usage

You can install the package directly from [`PYPI`](https://pypi.org/project/torch-frft/) using `pip` or `poetry` as follows:

```sh
pip install torch-frft
```

or

```sh
poetry add torch-frft
```

or directly from [`Conda`](https://anaconda.org/conda-forge/torch-frft) 

```sh
conda install -c conda-forge torch-frft
```

### For Development

This codebase utilizes [`Poetry`](https://python-poetry.org) for package management. To install the dependencies:

```sh
poetry install
```

or one can install the dependencies provided in [`requirements.txt`](requirements.txt) using `pip` or `conda`, e,g.,

```sh
pip install -r requirements.txt
```

## Usage

### Transforms

> [!WARNING]  
> Transforms applied in the same device as the input tensor. If the input tensor is on GPU, the transform will also be applied on GPU.

The package provides transform functions that operate on the $n^{th}$ dimension of an input tensor, `frft()` and `dfrft()`, which correspond to the fast computation of continuous fractional Fourier transform (FRFT) and discrete fractional Fourier transform (DFRFT), respectively. It also provides a function `dfrftmtx()`, which computes the DFRFT matrix for a given length and order, similar to MATLAB's `dftmtx()` function for the ordinary DFT matrix. Note that the `frft()` only operates on even-sized lengths as in the original MATLAB implementation [fracF.m](http://www.ee.bilkent.edu.tr/~haldun/fracF.m).

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

One can also place these layers directly into the `torch.nn.Sequential`. Remark that these transforms generate complex-valued outputs, so one may need to convert them to real-valued outputs, e.g., taking the real part, absolute value, etc. For example, the following code snippet implements a simple fully connected network with real parts of `FrFTLayer` and `DFrFTLayer` in between:

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

Creating custom layers with the provided `frft()` and `dfrft()` functions is also possible. The below example contains in-between `Linear` and `ReLU` layers and the same fractional order for forward and backward DFRFT transforms is as follows:

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

## FRFT Shift

Note that the fast computation of continuous FRFT is defined for the central grid of $\left[-\lfloor\frac{N}{2}\rfloor, \lfloor\frac{N-1}{2}\rfloor\right]$. Therefore, we need `fftshift()` to create equivalence with the original FFT when the transform order is precisely $1$. In this package, we also provide a shifted version of the fast FRFT computation, `frft_shifted()`, which operates with the assumption that the grid is $[0, N-1]$. The latter interval is not the default behavior since we want consistency with the original MATLAB implementation. The all there lines below are equivalent:

```python
import torch
from torch.fft import fft, fftshift
from torch_frft.frft_module import frft, frft_shifted

torch.manual_seed(0)
x = torch.rand(100)
y1 = fft(x, norm="ortho")
y2 = fftshift(frft(fftshift(x), 1.0))
y3 = frft_shifted(x, 1.0)

assert torch.allclose(y1, y2, atol=1e-5)
assert torch.allclose(y1, y3, atol=1e-5)
```
