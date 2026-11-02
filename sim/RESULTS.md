# Simulation results (n = 2000, R = 500 per distribution, alpha = 2 / xi = 0.5)

Figures: `fig_rmse_xi.png`, `fig_bias_xi.png`, `fig_rmse_q.png` (VaR_0.9995).
Full tables: `results_tables.txt`, `summary_table.csv`, `summary_by_k.csv`.
Raw per-sample estimates (both PH2 sub-family fits kept): `mc_results.csv.gz`.

## Design

| DGP | rho (tail-function scale) | in PH2 class |
|---|---|---|
| PH2-Erlang: log Y ~ .5 Exp(2) + .5 Erlang(2,2) | 0 | yes |
| Loggamma: log Y ~ Gamma(1.5, rate 2) | 0 | no |
| Burr: S = (1+y^0.5)^-4 | -0.5 | no |
| Two-Pareto: log Y ~ .3 Exp(2) + .7 Exp(2.5) | -0.5 | yes |
| GPD(xi=.5, sigma=1) | -1 | no |
| Frechet(2) | -2 | no |

Same k exceedances (k = 25..1000) for every method: Hill, GPD-MLE on Y-u,
PH2-MLE on log(Y/u) with sub-family (A) two exponentials (rho = -tau) and
(B) Exp + Erlang(2) (rho = 0); PH2 variants: AIC, BIC, A-only, B-only, Akaike-weighted.

## Main findings

1. **The PH2 Exp+Erlang sub-model (B) is the threshold-robust estimator.**
   Its RMSE(xi) curve is flat in k and it never shows GPD's small-k variance explosion
   (GPD RMSE 0.64-0.73 at k = 50) nor Hill's growing bias (GPD/Burr DGPs).
   Best-k RMSE(xi): B beats GPD and Hill on PH2-Erlang (0.039 vs 0.073 / 0.100),
   GPD (0.033 vs 0.044 / 0.088) and Burr (0.123 vs 0.187 / 0.260); ties GPD on Loggamma (0.050 vs 0.051).
   For VaR_0.9995, B has lower RMSE than GPD at every k <= 400 for all six DGPs.
2. **Hill wins for near-Pareto tails** (Two-Pareto, Frechet), as theory predicts
   when the second-order term is weak; B is second there, with a stable negative bias (about -0.06 to -0.09).
3. **The general two-exponential sub-family (A) hurts.** When AIC selects A,
   RMSE(xi) is 0.3-0.5 vs 0.04-0.13 when B is selected, even for the in-class Two-Pareto.
   The fitted second exponential models the body near the threshold, not the tail:
   median rho-hat from A is -5 to -33 against true values 0 to -2.
4. **rho is not identified at n = 2000.** AIC selects the rho = 0 sub-family 60-100 % of the
   time for every DGP (BIC 92-100 %), including rho = -0.5 and -2. The "PH2 tells which
   rho case you are in" claim is not supported at this sample size.
5. **Large k with mass near 0** (GPD, Burr, k >= 600): all log-scale methods break because
   log(Y/u) is dominated by the body; GPD-MLE does not. The JBF min-shift remedy was not applied here.

## Suggested framing for the paper

Present the Exp+Erlang PH2 (a Pareto-loggamma mixture, 2 parameters) as a
second-order-aware generalisation of Hill that competes with GPD at equal parameter count:
it extends Hill by a slowly varying log factor that absorbs second-order bias, so it
tolerates far lower thresholds than Hill and much smaller k than GPD.
Treat the two-eigenvalue case and rho estimation as a limitation / future work, or test it at
larger n (e.g. 10^4) with a parametric-bootstrap LRT for rho = 0 (boundary problem).
