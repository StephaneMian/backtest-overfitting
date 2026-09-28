"""Sharpe ratio inference.

Conventions: `returns` are per-period excess returns; Sharpe ratios are
per period unless stated otherwise (multiply by sqrt(periods per year) to
annualise). References:
  Lo (2002), "The Statistics of Sharpe Ratios";
  Mertens (2002) for the non-normal asymptotic variance;
  Bailey & Lopez de Prado (2012, 2014) for the PSR and the deflated SR.
"""
import numpy as np
from scipy.stats import norm, skew, kurtosis

EULER_GAMMA = 0.5772156649015329


def sharpe_ratio(returns, axis=0):
    r = np.asarray(returns, float)
    return r.mean(axis=axis) / r.std(axis=axis, ddof=1)


def sharpe_stderr(sr, n_obs, skewness=0.0, kurt=3.0):
    """Asymptotic standard error of the per-period Sharpe ratio.

    kurt is the (non-excess) kurtosis; skewness=0, kurt=3 gives the i.i.d.
    normal formula sqrt((1 + sr^2/2) / n).
    """
    var = (1.0 - skewness * sr + (kurt - 1.0) / 4.0 * sr**2) / (n_obs - 1)
    return np.sqrt(var)


def probabilistic_sharpe_ratio(returns, sr_benchmark=0.0):
    """P(true SR > sr_benchmark) given the observed track record (per period)."""
    r = np.asarray(returns, float)
    sr = sharpe_ratio(r)
    se = sharpe_stderr(sr, r.size, skew(r), kurtosis(r, fisher=False))
    return norm.cdf((sr - sr_benchmark) / se)


def expected_max_sharpe(n_trials, sr_std):
    """Approximate E[max of n_trials independent null Sharpe ratios].

    sr_std is the standard deviation of the Sharpe estimates across trials.
    Uses the extreme-value approximation of Bailey & Lopez de Prado (2014).
    """
    if n_trials < 2:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return sr_std * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe_ratio(returns, n_trials, sr_std):
    """PSR of the selected strategy against the expected maximum of the null trials."""
    return probabilistic_sharpe_ratio(returns, expected_max_sharpe(n_trials, sr_std))
