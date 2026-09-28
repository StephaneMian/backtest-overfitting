"""Reproduce every figure and number of the report.

Usage:  python experiments/run_experiments.py   (a few minutes on a laptop)
Outputs: report/figures/*.pdf and experiments/results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bto.sharpe import (sharpe_ratio, sharpe_stderr, probabilistic_sharpe_ratio,  # noqa: E402
                        expected_max_sharpe, deflated_sharpe_ratio)
from bto.multiple_testing import bonferroni, holm, benjamini_hochberg  # noqa: E402
from bto.pbo import cscv_pbo  # noqa: E402
from bto.simulate import strategy_returns, PERIODS_PER_YEAR  # noqa: E402

FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.figsize": (6.4, 4.0), "axes.grid": True, "grid.alpha": 0.3})
ANN = np.sqrt(PERIODS_PER_YEAR)
T = 5 * PERIODS_PER_YEAR  # five years of daily data
results = {}

# ------------------------------------------------ 1. The best of N noise strategies
print("1. Maximum Sharpe of null strategies")
Ns = [1, 10, 30, 100, 300, 1000]
reps = 200
rng = np.random.default_rng(100)
sim_mean, sim_q95 = [], []
for N in Ns:
    maxima = np.array([sharpe_ratio(strategy_returns(T, N, rng)).max() for _ in range(reps)]) * ANN
    sim_mean.append(float(maxima.mean()))
    sim_q95.append(float(np.quantile(maxima, 0.95)))
sr_sd = sharpe_stderr(0.0, T) * ANN
theory = [float(expected_max_sharpe(N, sr_sd)) if N > 1 else 0.0 for N in Ns]
results["max_sharpe"] = {"n_trials": Ns, "sim_mean": sim_mean, "sim_q95": sim_q95,
                         "theory_mean": theory, "years": 5}
fig, ax = plt.subplots()
ax.semilogx(Ns, sim_mean, "o-", label="simulated mean of the maximum")
ax.semilogx(Ns, sim_q95, "o:", label="simulated 95th percentile")
ax.semilogx(Ns[1:], theory[1:], "k--", label="extreme-value approximation")
ax.set_xlabel("number of strategies tried N")
ax.set_ylabel("annualised Sharpe of the best backtest")
ax.set_title("Five years of pure noise: the best backtest still looks good")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "max_sharpe.pdf")
plt.close(fig)

# ------------------------------------------------ 2. Is the selected strategy significant?
print("2. Naive PSR vs deflated SR on the selected strategy")
N, reps = 100, 400
rng = np.random.default_rng(200)
naive, deflated = [], []
for _ in range(reps):
    R = strategy_returns(T, N, rng)
    sr = sharpe_ratio(R)
    best = R[:, np.argmax(sr)]
    naive.append(probabilistic_sharpe_ratio(best) > 0.95)
    deflated.append(deflated_sharpe_ratio(best, N, sr.std(ddof=1)) > 0.95)
null_rates = {"naive_psr": float(np.mean(naive)), "deflated": float(np.mean(deflated))}

# power: one strategy has a true annual Sharpe, the other 99 are noise
power = {"true_sharpe": [], "naive_psr": [], "deflated": [], "selected_is_true": []}
for true_sr in [0.5, 1.0, 1.5, 2.0, 2.5]:
    hits_n, hits_d, correct = [], [], []
    for _ in range(200):
        R = strategy_returns(T, N, rng, annual_sharpe=np.r_[true_sr, np.zeros(N - 1)])
        sr = sharpe_ratio(R)
        j = int(np.argmax(sr))
        hits_n.append(probabilistic_sharpe_ratio(R[:, j]) > 0.95)
        hits_d.append(deflated_sharpe_ratio(R[:, j], N, sr.std(ddof=1)) > 0.95)
        correct.append(j == 0)
    power["true_sharpe"].append(true_sr)
    power["naive_psr"].append(float(np.mean(hits_n)))
    power["deflated"].append(float(np.mean(hits_d)))
    power["selected_is_true"].append(float(np.mean(correct)))
results["selection"] = {"n_trials": N, "null_rejection": null_rates, "power": power}
fig, ax = plt.subplots()
ax.plot(power["true_sharpe"], power["naive_psr"], "o-", label="naive PSR > 0.95")
ax.plot(power["true_sharpe"], power["deflated"], "s-", label="deflated SR > 0.95")
ax.plot(power["true_sharpe"], power["selected_is_true"], "^:", label="best backtest is the real signal")
ax.axhline(null_rates["naive_psr"], color="C0", lw=0.8, ls="--")
ax.axhline(null_rates["deflated"], color="C1", lw=0.8, ls="--")
ax.set_xlabel("true annualised Sharpe of the single real strategy")
ax.set_ylabel("frequency over simulations")
ax.set_title("Selecting the best of 100 backtests (dashed: rate under the null)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "dsr_power.pdf")
plt.close(fig)

# ------------------------------------------------ 3. Multiple testing on a universe of signals
print("3. Multiple-testing corrections")
N, n_true, true_sr, reps = 1000, 50, 1.0, 200
rng = np.random.default_rng(300)
methods = {"none (p < 0.05)": lambda p: p < 0.05, "Bonferroni": bonferroni,
           "Holm": holm, "Benjamini-Hochberg": benjamini_hochberg}
acc = {m: {"fdp": [], "power": [], "any_false": []} for m in methods}
truth = np.r_[np.ones(n_true, bool), np.zeros(N - n_true, bool)]
for _ in range(reps):
    R = strategy_returns(T, N, rng, annual_sharpe=np.where(truth, true_sr, 0.0))
    sr = sharpe_ratio(R)
    p = 1 - norm.cdf(sr / sharpe_stderr(sr, T))  # one-sided test of SR > 0
    for m, f in methods.items():
        rej = f(p)
        false = np.sum(rej & ~truth)
        acc[m]["fdp"].append(false / max(rej.sum(), 1))
        acc[m]["power"].append(np.sum(rej & truth) / n_true)
        acc[m]["any_false"].append(false > 0)
mt = {m: {"FDR": float(np.mean(v["fdp"])), "power": float(np.mean(v["power"])),
          "FWER": float(np.mean(v["any_false"]))} for m, v in acc.items()}
results["multiple_testing"] = {"n_strategies": N, "n_true": n_true, "true_sharpe": true_sr, "table": mt}

# ------------------------------------------------ 4. CSCV probability of overfitting
print("4. PBO")
rng = np.random.default_rng(400)
T_pbo = 4 * PERIODS_PER_YEAR
pbo = {}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for label, sr_vec in {"100 noise strategies": np.zeros(100),
                      "99 noise + 1 real (SR 2)": np.r_[2.0, np.zeros(99)]}.items():
    res = cscv_pbo(strategy_returns(T_pbo, 100, rng, annual_sharpe=sr_vec), n_blocks=12)
    pbo[label] = {"pbo": res.pbo, "median_is_sr": float(np.median(res.is_sharpe) * ANN),
                  "median_oos_sr": float(np.median(res.oos_sharpe) * ANN), "n_splits": int(res.logits.size)}
    axes[0].hist(res.logits, bins=40, alpha=0.6, density=True, label=f"{label} (PBO={res.pbo:.2f})")
    axes[1].scatter(res.is_sharpe * ANN, res.oos_sharpe * ANN, s=4, alpha=0.4, label=label)
axes[0].axvline(0, color="k", lw=0.8)
axes[0].set_xlabel(r"logit of OOS rank, $\lambda$")
axes[0].legend(fontsize=7)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xlabel("in-sample Sharpe of the selected strategy")
axes[1].set_ylabel("its out-of-sample Sharpe")
axes[1].legend(fontsize=7, markerscale=3)
fig.tight_layout()
fig.savefig(FIG / "pbo.pdf")
plt.close(fig)
# PBO is itself estimated from a single dataset: how much does it vary?
for label, sr_vec in {"100 noise strategies": np.zeros(100),
                      "99 noise + 1 real (SR 2)": np.r_[2.0, np.zeros(99)]}.items():
    vals = [cscv_pbo(strategy_returns(T_pbo, 100, rng, annual_sharpe=sr_vec), n_blocks=12).pbo
            for _ in range(20)]
    pbo[label]["pbo_over_20_datasets"] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                                           "min": float(np.min(vals)), "max": float(np.max(vals))}
results["pbo"] = pbo

# ------------------------------------------------ 5. Fat tails: confidence-interval coverage
print("5. Coverage of Sharpe confidence intervals under fat tails")
rng = np.random.default_rng(500)
cov = {}
n_obs, M, sr_true = PERIODS_PER_YEAR, 4000, 1.0
for dist in ["normal", "t"]:
    R = strategy_returns(n_obs, M, rng, annual_sharpe=np.full(M, sr_true), dist=dist, df=3.5)
    sr = sharpe_ratio(R)
    target = sr_true / ANN
    se_norm = sharpe_stderr(sr, n_obs)
    se_adj = sharpe_stderr(sr, n_obs, skew(R, axis=0), kurtosis(R, axis=0, fisher=False))
    cov[dist] = {"iid_normal_formula": float(np.mean(np.abs(sr - target) <= 1.96 * se_norm)),
                 "moment_adjusted_formula": float(np.mean(np.abs(sr - target) <= 1.96 * se_adj)),
                 "empirical_sd_over_normal_se": float(sr.std() / se_norm.mean())}
results["coverage"] = cov

(ROOT / "experiments" / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=1))
