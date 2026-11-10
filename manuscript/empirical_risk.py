"""Risk measures and excess-of-loss layer premiums for a loss data set, by threshold and method.

usage:  python3 empirical_risk.py LOSSES.csv COLUMN YEARS [OUTPREFIX]
  YEARS      length of the observation period in years (claim frequency = n / YEARS)
  OUTPREFIX  default figures/danish_risk

For each threshold (k largest observations) fits the Pareto (Hill), GPD and Exp+Erlang PH2
tails and computes ES_0.995, ES_0.999 and the expected annual loss in three XL layers
(closed forms in ../sim/riskmeasures.py). Writes OUTPREFIX.csv (all thresholds),
OUTPREFIX_ci.csv (95% profile-likelihood intervals at selected thresholds) and OUTPREFIX.png.
"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
from riskmeasures import fit_all, measure, profile_ci  # noqa: E402

LAYERS = [(20.0, 30.0), (50.0, 50.0), (100.0, 100.0)]   # (retention d, limit L): "L xs d"
CI_KS = (109, 246, 482)
METHODS = ("Hill", "GPD", "E+E")
COL = {"Hill": ("#2a78d6", "--", "s"), "GPD": ("#eb6834", "-.", "^"), "E+E": ("#1baf7a", "-", "o")}
LABEL = {"Hill": "Hill (Pareto)", "GPD": "GPD", "E+E": "PH$_2$ Exp+Erlang"}


def lname(d, L):
    return f"{L:g} xs {d:g}"


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    path, col, years = sys.argv[1], sys.argv[2], float(sys.argv[3])
    prefix = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, "figures", "danish_risk")
    y = pd.read_csv(path)[col]
    y = np.sort(pd.to_numeric(y, errors="coerce").dropna().values)
    y = y[y > 0]
    n = len(y)
    lam = n / years
    ks = np.unique(np.round(np.geomspace(max(20, n * 0.01), 0.5 * n, 30)).astype(int))
    specs = {"ES995": ("ES", 0.995), "ES999": ("ES", 0.999)}
    for d, L in LAYERS:
        specs[lname(d, L)] = ("layer", d, L)

    rows, cis = [], []
    for k in ks:
        f = fit_all(y, k)
        for mm in METHODS:
            mod, ll = f["models"][mm]
            row = dict(k=k, u=f["u"], method=mm)
            for key, spec in specs.items():
                if spec[0] == "layer" and spec[1] < f["u"]:
                    row[key] = np.nan          # retention below the threshold: not modelled
                    continue
                val = measure(mod, f["frac"], spec)
                row[key] = val * lam if spec[0] == "layer" else val
            rows.append(row)
            if k in CI_KS:
                for key, spec in specs.items():
                    if spec[0] == "layer" and spec[1] < f["u"]:
                        continue
                    vh = measure(mod, f["frac"], spec)
                    if not np.isfinite(vh):
                        continue
                    lo, hi = profile_ci(mm, f["ex"], f["u"], f["frac"], spec, vh, ll)
                    s = lam if spec[0] == "layer" else 1.0
                    cis.append(dict(k=k, u=f["u"], method=mm, measure=key, est=vh * s,
                                    lo=lo * s, hi=hi * s))
    df = pd.DataFrame(rows)
    df.to_csv(prefix + ".csv", index=False)
    ci = pd.DataFrame(cis)
    ci.to_csv(prefix + "_ci.csv", index=False)

    emp = {lname(d, L): np.clip(y - d, 0, L).sum() / years for d, L in LAYERS}
    pd.Series(emp).to_csv(prefix + "_burning_cost.csv", header=["burning_cost_per_year"])

    panels = ["ES995"] + [lname(d, L) for d, L in LAYERS]
    titles = {"ES995": "Expected shortfall $\\mathrm{ES}_{0.995}$"}
    titles.update({lname(d, L): f"Expected annual loss, layer {lname(d, L)}" for d, L in LAYERS})
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, key in zip(axs.flat, panels):
        for mm in METHODS:
            s = df[df.method == mm]
            c, ls, mk = COL[mm]
            ax.plot(s.k, s[key], ls, marker=mk, ms=4, lw=1.8, color=c, label=LABEL[mm])
        if key in emp:
            ax.axhline(emp[key], color="#888", lw=1, label="empirical burning cost")
        ax.set_xscale("log"); ax.set_title(titles[key], fontsize=10)
        ax.set_xlabel(f"k = number of exceedances (n = {n})")
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axs[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(prefix + ".png", dpi=150)

    pd.set_option("display.width", 250)
    print(f"n = {n}, claim frequency = {lam:.1f} per year; burning cost per year: "
          + ", ".join(f"{a}: {b:.3f}" for a, b in emp.items()))
    print(df.round(3).to_string(index=False))
    print(ci.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
