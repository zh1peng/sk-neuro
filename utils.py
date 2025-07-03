"""Miscellaneous helper functions."""

import numpy as np
from scipy.stats import pearsonr


def correlation(x: np.ndarray, y: np.ndarray):
    """Return Pearson r and p-value."""
    return pearsonr(x, y)


def bootstrap_corr(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, random_state: int | None = None):
    """Bootstrap confidence interval for the correlation."""
    rng = np.random.RandomState(random_state)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        stats.append(pearsonr(x[idx], y[idx])[0])
    return np.percentile(stats, [2.5, 97.5])

