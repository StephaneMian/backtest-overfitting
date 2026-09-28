"""Synthetic strategy returns with a known ground truth."""
import numpy as np

PERIODS_PER_YEAR = 252


def strategy_returns(n_obs, n_strategies, rng, annual_sharpe=None, vol=0.01,
                     dist="normal", df=4):
    """T x N daily returns. annual_sharpe: array of true annualised Sharpe
    ratios (default all zero). dist="t" draws Student-t shocks rescaled to
    unit variance, to test the effect of fat tails."""
    sr = np.zeros(n_strategies) if annual_sharpe is None else np.asarray(annual_sharpe, float)
    if dist == "normal":
        eps = rng.standard_normal((n_obs, n_strategies))
    elif dist == "t":
        eps = rng.standard_t(df, (n_obs, n_strategies)) / np.sqrt(df / (df - 2))
    else:
        raise ValueError(dist)
    mu = sr / np.sqrt(PERIODS_PER_YEAR) * vol
    return mu + vol * eps
