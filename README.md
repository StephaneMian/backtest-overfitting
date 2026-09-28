# When is a backtest real?

Selection bias in strategy research, measured on synthetic strategies whose true Sharpe ratio is known: the Sharpe ratio of the best of N noise backtests, the probabilistic and deflated Sharpe ratios, multiple-testing corrections (Bonferroni, Holm, Benjamini–Hochberg) and the probability of backtest overfitting (CSCV).

The full write-up is in [`report/report.pdf`](report/report.pdf).

## Main results (five years of daily data)

| Question | Result |
|---|---|
| Annualised Sharpe of the best of N zero-Sharpe strategies | 1.11 for N = 100, 1.44 for N = 1000; extreme-value formula within 5 % |
| Best of 100 noise strategies declared significant | 99.5 % with the naive PSR, 0.5 % with the deflated Sharpe ratio |
| One real strategy among 100 | detected by the deflated SR in 1.5 % of cases at Sharpe 1.0, 55 % at Sharpe 2.0 |
| 1000 strategies, 50 real (Sharpe 1.0) | no correction: 56 % of discoveries are false; Benjamini–Hochberg: FDR 5.6 %, power 13 % |
| PBO across 20 datasets | 0.47 ± 0.15 for pure noise, 0.23 ± 0.19 with one real strategy |

## Layout

```
src/bto/
  sharpe.py             Sharpe ratio, standard error (non-normal), PSR, expected maximum, deflated SR
  multiple_testing.py   Bonferroni, Holm, Benjamini–Hochberg
  pbo.py                combinatorially symmetric cross-validation
  simulate.py           synthetic strategy returns with known Sharpe ratios
experiments/run_experiments.py   produces every figure and number of the report
tests/                           checks against simulation and textbook examples
report/                          LaTeX source, figures and PDF
```

## Reproduce

```bash
pip install -e ".[test]"
pytest -q
python experiments/run_experiments.py      # a few minutes, writes report/figures and experiments/results.json
cd report && pdflatex report.tex && pdflatex report.tex
```
