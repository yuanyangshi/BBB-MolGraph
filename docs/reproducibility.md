# Scientific Reproducibility Protocol

BBB-MolGraph is designed from the ground up to comply with the most stringent reproducibility guidelines established by leading journals (e.g., ACS JCIM, Nature Machine Intelligence, Briefings in Bioinformatics).

## 1. Deterministic Seeds
Pseudo-random number generators across Python, NumPy, PyTorch CPU, and CUDA cuDNN are locked at the framework level:
```python
from bbb_molgraph.utils.seed import set_deterministic_seed
set_deterministic_seed(42)
```
Environment variables configured:
- `PYTHONHASHSEED = "42"`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `CUBLAS_WORKSPACE_CONFIG = ":4096:8"`

## 2. Fixed Scaffold Splits
To prevent data leakage caused by random splitting of closely related chemical derivatives, we enforce **Bemis-Murcko Scaffold Splitting**.
- Splits are precomputed and frozen in `data/splits/scaffold_folds.json`.
- Any researcher running `examples/02_run_scaffold_cv.py` evaluates on the exact same folds.

## 3. 95% Bootstrap Confidence Intervals
Metrics are not reported as single lucky numbers. In accordance with top-tier editorial criteria, all key metrics are evaluated over 1,000 non-parametric bootstrap resamples to establish standard deviations and 95% confidence intervals:
$$CI_{95\%} = [\text{Percentile}_{2.5\%}, \text{Percentile}_{97.5\%}]$$
