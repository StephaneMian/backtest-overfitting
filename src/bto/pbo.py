"""Probability of backtest overfitting via combinatorially symmetric
cross-validation (CSCV), Bailey, Borwein, Lopez de Prado & Zhu (2017).

The T x N matrix of strategy returns is cut into S contiguous blocks. For each
of the C(S, S/2) ways to pick half of the blocks as in-sample, the strategy
with the best in-sample Sharpe is selected and its out-of-sample rank is
recorded. PBO is the fraction of splits where that strategy ranks below the
out-of-sample median.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np

from .sharpe import sharpe_ratio


@dataclass
class PBOResult:
    pbo: float
    logits: np.ndarray
    is_sharpe: np.ndarray    # in-sample SR of the selected strategy, per split
    oos_sharpe: np.ndarray   # its out-of-sample SR, per split


def cscv_pbo(returns, n_blocks=16):
    R = np.asarray(returns, float)
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    T, N = R.shape
    blocks = np.array_split(np.arange(T), n_blocks)
    logits, is_sr, oos_sr = [], [], []
    for chosen in combinations(range(n_blocks), n_blocks // 2):
        in_idx = np.concatenate([blocks[i] for i in chosen])
        out_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in chosen])
        sr_in = sharpe_ratio(R[in_idx])
        sr_out = sharpe_ratio(R[out_idx])
        best = int(np.argmax(sr_in))
        # relative rank in (0, 1): 1 = best out of sample
        rank = (np.sum(sr_out < sr_out[best]) + 0.5 * (np.sum(sr_out == sr_out[best]) - 1) + 1) / (N + 1)
        logits.append(np.log(rank / (1 - rank)))
        is_sr.append(sr_in[best])
        oos_sr.append(sr_out[best])
    logits = np.array(logits)
    return PBOResult(float(np.mean(logits <= 0)), logits, np.array(is_sr), np.array(oos_sr))
