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
   For VaR_0.9995, B has lower RMSE than GPD at every k <= 400 for five of the six DGPs
   (for Burr only up to k = 200).
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

---

# Rerun: n = 10000, R = 500, with JBF min-shift and GPD-loc5 (`n10000/`)

Changes: n = 10000; k grid scaled by 5 (125..5000); VaR level p = 1 - 1/n = 0.9999;
every sample also fitted on the min-shifted scale log((Y-m)/(u-m)), m = min Y
(GPD on Y-u is shift-invariant); new DGP GPD-loc5 (GPD xi = .5, sigma = 1, location 5, as in JBF Fig. 3).
Figures: `n10000/fig_rmse_xi.png`, `fig_bias_xi.png`, `fig_rmse_q.png` (unshifted methods) and
`n10000/fig_shift_rmse_xi.png`, `fig_shift_rmse_q.png` (unshifted vs min-shifted).

Best-k RMSE(xi), k in 250..5000:

| DGP | PH2 Exp+Erlang | PH2 Exp+Erlang, shift | GPD | Hill |
|---|---|---|---|---|
| PH2-Erlang (rho 0) | **0.016** | 0.027 | 0.062 | 0.081 |
| Loggamma (rho 0) | 0.034 | 0.017¹ | **0.028** | 0.054 |
| Burr (rho -.5) | **0.086** | 0.086 | 0.132 | 0.242 |
| Two-Pareto (rho -.5) | 0.078 | 0.013¹ | 0.057 | 0.058 |
| GPD (rho -1) | **0.018** | 0.018 | 0.020 | 0.067 |
| GPD-loc5 (rho -1) | 0.096 | **0.019** | 0.021 | 0.074 |
| Frechet (rho -2) | 0.064 | 0.043 | 0.038 | **0.021** |

¹ The shifted estimator's bias passes through zero as k grows (it is negative at small k and
positive at large k), so its best-k RMSE is partly an artefact of that crossing; its RMSE curve
is not flat.

Findings:

1. **The n = 2000 conclusions hold and sharpen.** The Exp+Erlang PH2 sub-model has the flattest
   RMSE curve, and its RMSE keeps falling with k for rho = 0 DGPs (0.016 at k = 5000 on PH2-Erlang,
   versus GPD 0.062 and Hill 0.160). It ties GPD on GPD data and beats both on Burr. On near-Pareto tails it loses:
   Hill and GPD tie on Two-Pareto (0.058 / 0.057 vs 0.078) and Hill wins on Frechet.
2. **The min-shift does what it is designed for and nothing more.** It is a no-op on GPD and Burr
   (support from 0); the large-k breakdown there remains (it comes from log of body values near 0).
   On GPD-loc5 it is essential: unshifted PH2-B and Hill have a bias of -0.2 at k = 5000; shifted
   PH2-B matches GPD (0.019 vs 0.021) for k <= 3000. On support-at-1 DGPs (PH2-Erlang, Loggamma,
   Two-Pareto) the shift distorts the large-k fits (log(Y - 1) is not PH2), and Hill with the shift
   degrades badly. The shift should be applied only when the support offset is real (JBF's advice).
3. **rho is still not identified at n = 10000.** AIC picks rho = 0 in 50-100 % of fits for every
   DGP except unshifted GPD-loc5 (Two-Pareto 50-70 %, Burr 92-100 %; BIC 95-100 %).
   When the two-exponential sub-family is picked, the median rho-hat at k = 5000 is -2.3 to -3.4
   across DGPs with true values 0, -0.5, -1 and -2 (close to the truth only for Frechet). The one case where AIC moves decisively to A is
   unshifted GPD-loc5, where A is absorbing the support offset rather than a second-order term.
   On the min-shifted scale, selection goes to rho = 0 almost always (AIC 74-100 %).

Implication for the paper: frame PH2 as a robust, second-order-tolerant tail estimator via the
Exp+Erlang sub-model (applied with the min-shift only when the support starts away from 0);
do not claim that PH2 identifies the second-order case at realistic sample sizes.
