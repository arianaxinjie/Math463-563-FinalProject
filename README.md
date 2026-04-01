# MATH 463/563 —  Final Project

**Public interface:** use the **`core_code`** package and the unified driver **`optsolve`**.  
If you are **grading or reusing this code**, you only need the sections below through **Hyperparameters (`OptParams`)**.

The `scripts/` folder is **for the project authors** (local demos and per-algorithm tuning); it is **not** part of the documented API for external readers.

---

## Requirements

- Python 3.10+ recommended  

```bash
pip install numpy scipy matplotlib pillow scikit-image
```

`pillow` helps load images from files; parts of the stack work with reduced dependencies — see `core_code` imports.

## Install / import

Run Python with the **repository root** (the directory that contains `core_code/`) on `sys.path`, for example:

```bash
cd /path/to/this/repository
export PYTHONPATH="$(pwd):$PYTHONPATH"   # if needed
python your_driver.py
```

Then:

```python
import core_code as cc
```

---

## `optsolve` — unified solver (course-style API)

Signature:

```text
x = optsolve(problem, method, x0, kernel, b, params)
```

| Argument | Meaning |
|----------|---------|
| `problem` | `"l1"` or `"l2"` — data fidelity in the TV model |
| `method` | Algorithm name (string, case-insensitive); see table below |
| `x0` | Warm-start image, same shape as `b`, or **`None`** for default initialization |
| `kernel` | 2D blur kernel (summed to 1) |
| `b` | 2D observed image |
| `params` | `cc.OptParams(...)` or a `dict` (supports key `gammal1` as in the project PDF) |

**Return value:** restored image `x` (shape matches `b`). With `return_all=True`: `(x, obj_history, info)`.

### Example (minimal)

```python
import core_code as cc
import numpy as np

# You must supply kernel and b (e.g. from simulate_blur below)
x0 = None
p = cc.OptParams(
    maxiter=500,
    gamma=0.049,          # TV weight; PDF often calls this gammal1
    tprimaldr=2.0,
    rhoprimaldr=0.1,
)
x = cc.optsolve("l1", "douglasrachfordprimal", x0, kernel, b, p)
```

### Method names (`method`)

Examples (several aliases exist — full list in `core_code/optsolve.py`):

| Algorithm | Example `method` string |
|-----------|-------------------------|
| 1 — Primal Douglas–Rachford | `douglasrachfordprimal` |
| 2 — Primal–Dual DR | `douglasrachfordprimaldual` |
| 3 — ADMM | `admm` |
| 4 — Chambolle–Pock | `chambollepock` or `cp` |

### Building `kernel` and `b` from a clean image

Use **`core_code`** only (no `scripts/` required):

```python
import core_code as cc

img = cc.load_image("cameraman.jpg")   # or pass a NumPy array
kernel, x_blur, b = cc.generate_blurred_noisy_cfg(
    x_true=img,
    kernel_kind="gaussian",
    kernel_size=15,
    kernel_sigma=3.0,
    noise_type="gaussian",
    noise_sigma=0.001,
    mode="periodic",
)
x = cc.optsolve("l2", "admm", None, kernel, b, cc.OptParams(maxiter=300, verbose=False))
```

### Hyperparameters (`OptParams`)

Defaults and fields are defined in `core_code/optsolve.py` (`OptParams` dataclass). Common fields:

- `maxiter`, `tol`, `verbose`, `gamma` (alias `gammal1` in dicts)
- Primal DR: `tprimaldr`, `rhoprimaldr`
- Primal–Dual DR: `tprimaldualdr`, `rhoprimaldualdr`
- ADMM: `tadmm`, `rhoadmm`
- Chambolle–Pock: set both `tcp` and `scp` for fixed steps, or use `cp_step_theta` for automatic `t`, `s`

Merge a plain dict with defaults: `cc.merge_params({...})`.

### Quick terminal smoke test

```bash
cd /path/to/this/repository
python -c "
import numpy as np, core_code as cc
b = np.random.rand(32, 32)
k = np.ones((5,5)); k /= k.sum()
p = cc.OptParams(maxiter=20, verbose=False, tol=1e-2)
x = cc.optsolve('l2', 'douglasrachfordprimal', None, k, b, p)
print('OK', x.shape)
"
```

---

## Repository layout (for reference)

| Path | Role |
|------|------|
| `core_code/` | **Public package**: algorithms, `optsolve`, `build_problem`, `generate_blurred_noisy_cfg`, `load_image`, `show_results`, … |
| `scripts/` | **Internal** — authors’ convenience runners only (not required to use `optsolve`) |
| `testimages/` | Sample images for experiments |
| `comp463finalproject.py` | Legacy Colab export (reference) |

---

## For project authors only (`scripts/`)

The team uses `scripts/run_algorithm1.py` … `run_algorithm4.py` plus `run_common.py` to load `testimages/`, tweak per-person `HYPERPARAMETERS`, and save figures under `output/`. **Graders and external users can ignore this directory** and call `optsolve` as above.
