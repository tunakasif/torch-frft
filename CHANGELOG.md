## 0.4.0 (2023-08-03)

### Feat

- update sample training notebook
- **layer**: add discrete transform version
- add inverse wrappers for transforms
- **ndim**: add `n`-dimensional `dfrft`
- implement `dfrft` matrix
- **dfrft**: implement `dis_s()` helper function
- **discrete**: implement `P` construction
- initialize `dfrft` implementation

### Fix

- allow `Tensor` type for `dfrft` order
- **dfrft**: `ndim` implementation
- **layer**: add dimension to `FrFT`

### Refactor

- filenames due to function names
- **dfrft**: update naming convention
- rename `fracF` to `frft`
- change `trainable-frft` to `torch-frft`

## 0.3.0 (2023-07-31)

### Feat

- **ndim**: increase `ndim` compatibility
- **ndim**: start converting `fracF()` to supply `dim` selection
- implement `.mul()` with `nth` dimension
- initialize trainable module
- convert `bizdec()` & `bizinter()` to `ndim`

### Fix

- dimension check in `einstr` generation
- **ndim**: `corefrmod2()` for n-dimension
- with `ndim` update `fracF()` diff'able
- 1D tensor from `(N,1)` to `(N,)`
- remove `jax` based `dfrt` implementation

## 0.2.0 (2023-07-27)

### Feat

- update `fracF()` for integer values
- **torch**: initialize `fracF` implementation
- **dfrt**: start importing functions for `dfrt`

### Fix

- incorrect calculation of `corefrmod2()` (#2)

## 0.1.0 (2023-06-07)

### Feat

- **init**: trainable `frft` implementation

### Fix

- **commitizen**: project name and version
