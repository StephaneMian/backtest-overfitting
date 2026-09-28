import numpy as np
import pytest

from bto.sharpe import (sharpe_ratio, sharpe_stderr, probabilistic_sharpe_ratio,
                        expected_max_sharpe, deflated_sharpe_ratio)
from bto.multiple_testing import bonferroni, holm, benjamini_hochberg
from bto.pbo import cscv_pbo
from bto.simulate import strategy_returns


def test_stderr_reduces_to_normal_formula():
    assert sharpe_stderr(0.1, 1001) == pytest.approx(np.sqrt((1 + 0.1**2 / 2) / 1000))


def test_stderr_matches_simulation():
    rng = np.random.default_rng(0)
    R = strategy_returns(500, 4000, rng, annual_sharpe=np.full(4000, 1.0))
    sr = sharpe_ratio(R)
    assert sr.std() == pytest.approx(sharpe_stderr(sr.mean(), 500), rel=0.05)


def test_expected_max_matches_simulation():
    rng = np.random.default_rng(1)
    maxima = rng.standard_normal((5000, 100)).max(axis=1)
    assert expected_max_sharpe(100, 1.0) == pytest.approx(maxima.mean(), rel=0.03)


def test_psr_is_uniform_under_the_null():
    rng = np.random.default_rng(2)
    R = strategy_returns(750, 2000, rng)
    psr = np.array([probabilistic_sharpe_ratio(R[:, j]) for j in range(R.shape[1])])
    assert np.mean(psr > 0.95) == pytest.approx(0.05, abs=0.015)


def test_dsr_below_psr_for_selected_strategy():
    rng = np.random.default_rng(3)
    R = strategy_returns(750, 200, rng)
    sr = sharpe_ratio(R)
    best = R[:, np.argmax(sr)]
    assert deflated_sharpe_ratio(best, 200, sr.std()) < probabilistic_sharpe_ratio(best)


def test_corrections_on_known_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert bonferroni(p).sum() == 1
    assert holm(p).sum() == 1
    assert benjamini_hochberg(p).sum() == 2
    # Holm is never less powerful than Bonferroni
    rng = np.random.default_rng(4)
    for _ in range(100):
        q = rng.uniform(0, 0.1, 30)
        assert holm(q).sum() >= bonferroni(q).sum()


def test_pbo_high_for_noise_low_for_real_signal():
    rng = np.random.default_rng(5)
    noise = strategy_returns(1000, 50, rng)
    assert cscv_pbo(noise, n_blocks=8).pbo > 0.3
    real = strategy_returns(1000, 50, rng, annual_sharpe=np.r_[3.0, np.zeros(49)])
    assert cscv_pbo(real, n_blocks=8).pbo < 0.1
