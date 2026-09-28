"""Family-wise and false-discovery-rate corrections. Each returns a boolean
rejection mask aligned with the input p-values."""
import numpy as np


def bonferroni(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    return p <= alpha / p.size


def holm(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    thresholds = alpha / (m - np.arange(m))
    passed = p[order] <= thresholds
    # step-down: stop at the first failure
    k = m if passed.all() else int(np.argmin(passed))
    reject = np.zeros(m, bool)
    reject[order[:k]] = True
    return reject


def benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    below = p[order] <= alpha * np.arange(1, m + 1) / m
    reject = np.zeros(m, bool)
    if below.any():
        k = int(np.max(np.nonzero(below)[0])) + 1  # step-up: largest index that passes
        reject[order[:k]] = True
    return reject
