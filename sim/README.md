# Threshold-robustness simulation: PH2 vs Hill vs GPD

Monte Carlo study for the follow-up to Kim & Kim (2015, JBF).

* `ph2sim.py` – data-generating processes (all alpha = 2) and estimators
  (Hill, GPD MLE on Y-u, PH2 MLE on log(Y/u) with both sub-families:
  (A) two exponentials -> rho = -tau, (B) Exp + Erlang(2) -> rho = 0).
* `run_mc.py R` – runs R replications per distribution (n = 2000, k grid 25..1000),
  writes `mc_results.csv` with both PH2 sub-family fits per sample.
* `analyze.py` – builds PH2 variants (AIC, BIC, A-only, B-only, Akaike-weighted average),
  RMSE/bias curves vs k, summary tables, and rho-identification diagnostics.

```bash
pip install numpy scipy pandas matplotlib
python3 run_mc.py 500
python3 analyze.py
```
* `RESULTS.md` – summary of the n = 2000, R = 500 run and suggested framing.
