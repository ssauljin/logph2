"""Empirical illustration: Hill, GPD and Exp+Erlang PH2 tail-index estimates over thresholds.

usage:  python3 empirical.py LOSSES.csv [COLUMN] [OUTPREFIX]
  LOSSES.csv  a CSV file with one column of positive losses (header optional)
  COLUMN      column name or 0-based index (default: first column)
  OUTPREFIX   prefix for the output figure/table (default: figures/empirical)

Produces OUTPREFIX.png (xi-hat against k, with 95% profile-likelihood band for the
Exp+Erlang PH2 fit) and OUTPREFIX.csv (all estimates, the AIC sub-family choice, VaR).
Use the min-shift (--shift) only when the support of the losses starts away from zero.
"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
from ph2sim import fit_gpd, fit_ph2_both, ph2_quantile  # noqa: E402
from fisher_B import profile_ci_xi  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    shift = "--shift" in sys.argv
    if not args:
        sys.exit(__doc__)
    raw = pd.read_csv(args[0], header=None)
    if raw.iloc[0].astype(str).str.match(r"^[A-Za-z]").any():
        raw = pd.read_csv(args[0])
    col = args[1] if len(args) > 1 else 0
    y = raw[col] if col in raw.columns else raw.iloc[:, int(col)]
    y = np.sort(pd.to_numeric(y, errors="coerce").dropna().values)
    y = y[y > 0]
    n = len(y)
    m = y[0] if shift else 0.0
    prefix = args[2] if len(args) > 2 else os.path.join(HERE, "figures", "empirical")
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    p_var = 0.995
    ks = np.unique(np.round(np.geomspace(max(20, n * 0.01), 0.5 * n, 30)).astype(int))
    rows = []
    for k in ks:
        u = y[n - k - 1]
        ex = y[n - k:]
        x = np.log((ex - m) / (u - m))
        qq = n * (1 - p_var) / k
        xiH = x.mean()
        aG, xiG, qfG = fit_gpd(ex - u)
        f = fit_ph2_both(x)
        muB, _, nllB, sB = f["B"]
        thA, rhoA, nllA, sA = f["A"]
        lo, hi = profile_ci_xi(x, muB)
        rows.append(dict(k=k, u=u, xi_hill=xiH, xi_gpd=xiG, xi_ph2_EE=1 / muB,
                         xi_ph2_EE_lo=lo, xi_ph2_EE_hi=hi, xi_ph2_A=1 / thA, rho_A=rhoA,
                         aic_picks_rho0=2 * nllB + 4 <= 2 * nllA + 6,
                         var_hill=u * qq ** (-xiH), var_gpd=u + qfG(qq) if qq < 1 else np.nan,
                         var_ph2_EE=m + (u - m) * np.exp(ph2_quantile(sB, qq))))
    df = pd.DataFrame(rows)
    df.to_csv(prefix + ".csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(df.k, df.xi_ph2_EE_lo, df.xi_ph2_EE_hi, color="#1baf7a", alpha=0.18,
                    label="PH$_2$ Exp+Erlang, 95% profile CI")
    ax.plot(df.k, df.xi_hill, "--s", ms=4, color="#2a78d6", label="Hill")
    ax.plot(df.k, df.xi_gpd, "-.^", ms=4, color="#eb6834", label="GPD")
    ax.plot(df.k, df.xi_ph2_EE, "-o", ms=4, color="#1baf7a", label="PH$_2$ Exp+Erlang")
    ax.set_xscale("log"); ax.set_xlabel(f"k = number of exceedances (n = {n})")
    ax.set_ylabel(r"$\hat\xi = 1/\hat\alpha$")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(prefix + ".png", dpi=150)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
