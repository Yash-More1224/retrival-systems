"""Q4.4 -- Bootstrap 95% CI, shared by Q2/Q3 recall@K reporting and the
Q4 evaluation harness.

Bugs fixed relative to the original v1 draft (see SPEC.md Q4.4):
  - seeded (np.random.default_rng(seed)), so results are reproducible
  - resamples per-impression METRIC VALUES (the independent unit), not raw
    records -- callers pass one float per impression, already computed
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(per_impression_values: list[float], n_boot: int = 1000, ci: float = 0.95,
                  seed: int = 42) -> tuple[float, float, float]:
    """Returns (mean, ci_lower, ci_upper) over the non-None values in per_impression_values."""
    rng = np.random.default_rng(seed)
    x = np.asarray([v for v in per_impression_values if v is not None], dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot_means = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(x.mean()), float(lo), float(hi)
