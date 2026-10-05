# NumPy API & Module Structure

> Measured on a local NumPy 2.2.6 install + the official docs on [Module structure](https://numpy.org/doc/stable/reference/module_structure.html), compiled 2026-06-07

## 1. API Counts

| Dimension | Count |
|---|---|
| Top-level public API (`np.__all__`) | **499** |
| of which: functions / callables | 392 (including 106 ufuncs) |
| of which: classes (e.g. `ndarray`, `dtype`) | 70 |
| of which: constants (e.g. `pi`, `e`, `inf`) | 18 |
| Public `ndarray` methods / attributes | 74 |

### API count per submodule

| Submodule | APIs | Submodule | APIs |
|---|---|---|---|
| `np.ma` (masked arrays) | 226 | `np.linalg` | 32 |
| `np.random` | 60 | `np.fft` | 18 |
| `np.char` | 53 | `np.polynomial` | 13 |
| `np.testing` | 49 | `np.emath` | 9 |
| `np.strings` | 45 | `np.rec` | 9 |
| `np.dtypes` | 33 | `np.exceptions` | 6 |

### Takeaways

- Top-level API ≈ **500**
- Including submodules, about **1,070** (with overlap: `np.ma` mirrors many top-level functions)
- Adding `ndarray` methods and class methods such as `random.Generator`, the full API surface is about **1,200+**
- The common claim "NumPy has about 600 public functions" = deduplicated top-level functions + functions of the main submodules

## 2. Official Module Structure (20 public namespaces in three tiers)

### ① Main namespaces: recommended for everyday use (9)

| Module | Purpose |
|---|---|
| `numpy` | Main namespace: array creation, math, indexing and other core features |
| `numpy.exceptions` | NumPy-specific exceptions and warnings |
| `numpy.fft` | Fast Fourier transforms |
| `numpy.linalg` | Linear algebra (inverse, eigenvalues, SVD…) |
| `numpy.polynomial` | Polynomial representation and arithmetic |
| `numpy.random` | Random number generation, probability distributions |
| `numpy.strings` | String array operations (new in 2.0, replaces `np.char`) |
| `numpy.testing` | Test assertion helpers (`assert_allclose`, etc.) |
| `numpy.typing` | Type annotation support (`NDArray`, `ArrayLike`) |

### ② Special-purpose namespaces (6)

| Module | Purpose |
|---|---|
| `numpy.ctypeslib` | Interop with C libraries / ctypes |
| `numpy.dtypes` | Definitions of dtype classes (rarely used directly) |
| `numpy.emath` | Math functions that switch domain automatically (`sqrt(-1)` returns a complex number instead of nan) |
| `numpy.lib` | Miscellaneous utilities that don't fit the main namespace |
| `numpy.rec` | Record arrays (largely superseded by pandas) |
| `numpy.version` | Detailed version information |

### ③ Legacy namespaces: not recommended for new code (5)

| Module | Status |
|---|---|
| `numpy.char` | Legacy fixed-width string features → use `np.strings` instead |
| `numpy.distutils` | Deprecated build system (removed from Python 3.12 on) |
| `numpy.f2py` | Fortran binding generator, usually only used from the command line |
| `numpy.ma` | Masked arrays; the docs themselves call it "not very reliable, needs a rewrite" |
| `numpy.matlib` | For the `matrix` class, pending deprecation |

### About the "30 submodules" claim

The install directory actually contains **34** submodules/subpackages, but the extra ones are not public API:

- **Private modules** (11): `_core`, `_typing`, `_utils` etc. Leading underscore = internal implementation that may change at any time
- **Compatibility shells**: `core`, `compat`, `matrixlib`. Since 2.0 they are just aliases of `_core`, and importing them raises a deprecation warning
- **Non-API files**: `conftest`, `tests`, `__config__` and other test / build artifacts

Rule of thumb: **the 20 public ones are what matter to users; in daily work you mostly need `numpy` itself + `fft` / `linalg` / `random` / `polynomial`**.

## 3. `numpy.polynomial` in Detail

Treats polynomials as **objects**: define, evaluate, find roots, differentiate, integrate, combine and fit, each in one line.

### Core usage

```python
from numpy.polynomial import Polynomial

p = Polynomial([1, 2, 3])   # coefficients from low to high degree → p(x) = 1 + 2x + 3x²
p(2)          # evaluate → 17.0
p.roots()     # roots (may be complex)
p.deriv()     # derivative → 2 + 6x
p.integ()     # integral → x + x² + x³
p * q         # polynomial multiplication with plain operators
```

### Most common use: curve fitting

```python
fit = Polynomial.fit(x, y, deg=2)   # fit a quadratic to data points
fit.convert()                        # convert back to standard coefficients
```

Measured: fitting noisy `0.5x² - 2x + 3` data gives `0.52x² - 2.15x + 3.12`.

### Six polynomial classes

| Class | Basis | Use |
|---|---|---|
| `Polynomial` | Plain power series 1, x, x² | Enough for everyday work |
| `Chebyshev` | Chebyshev polynomials | Standard choice for numerical approximation; no oscillation at interval ends |
| `Legendre` | Legendre polynomials | Physics, spherical harmonics |
| `Hermite` / `HermiteE` | Hermite polynomials | Quantum mechanics, probability |
| `Laguerre` | Laguerre polynomials | Radial equation of the hydrogen atom, etc. |

Orthogonal polynomials matter in **numerical integration (Gaussian quadrature) and function approximation**: `Chebyshev.fit()` is far more numerically stable than a plain power series when fitting high-degree polynomials.

### Historical note

Older code often uses `np.polyfit()` / `np.poly1d`, the legacy API (coefficients ordered from high to low degree, i.e. reversed). The docs now recommend `numpy.polynomial`; don't use `poly1d` in new code.
