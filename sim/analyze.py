"""Analyse a Monte Carlo results file.

usage: python3 analyze.py [RESULTS] [OUTDIR] [N]
  defaults: mc_results.csv(.gz)  .  2000
VaR level is p = 1 - 1/N (0.9995 for N = 2000, 0.9999 for N = 10000).
If the file contains min-shifted fits (columns ending in _s), shifted variants are analysed too.
"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ph2sim import DISTS, ALPHA, true_quantile

RES = sys.argv[1] if len(sys.argv) > 1 else (
    "mc_results.csv" if os.path.exists("mc_results.csv") else "mc_results.csv.gz")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "."
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
os.makedirs(OUTDIR, exist_ok=True)
out = lambda f: os.path.join(OUTDIR, f)

P = 1 - 1 / N
XI = 1 / ALPHA
CLIP = 3.0     # cap |log(Qhat/Q)|; share of capped values is reported
XICAP = 5.0    # cap xi-hat (alpha-hat -> 0 explosions)
df = pd.read_csv(RES)
order = [d for d in DISTS if d in set(df.dist)]
df["qt"] = df["dist"].map({d: true_quantile(d, P) for d in order})
SHIFT = "aH_s" in df.columns


def xi_of(a):
    a = np.asarray(a, float)
    return np.clip(np.where(np.isfinite(a) & (a > 0), 1 / a, XICAP), None, XICAP)


def lq_of(q):
    q = np.asarray(q, float)
    o = np.log(np.clip(q, 1e-300, None) / df.qt.values)
    o[~np.isfinite(o)] = CLIP
    return np.clip(o, -CLIP, CLIP)


def ph2_variants(sfx, tag):
    """PH2 estimators built from the two stored sub-family fits (suffix '' or '_s')."""
    nA, nB, k = df["nllA" + sfx], df["nllB" + sfx], df.k
    aicB = 2 * nB + 4 <= 2 * nA + 6
    bicB = 2 * nB + 2 * np.log(k) <= 2 * nA + 3 * np.log(k)
    xA, xB = xi_of(df["aA" + sfx]), xi_of(df["aB" + sfx])
    lA, lB = lq_of(df["qA" + sfx]), lq_of(df["qB" + sfx])
    wB = 1 / (1 + np.exp(-((2 * nA + 6) - (2 * nB + 4)) / 2))
    df["selAIC_B" + sfx], df["selBIC_B" + sfx] = aicB, bicB
    return {
        f"PH2-AIC{tag}": (np.where(aicB, xB, xA), np.where(aicB, lB, lA)),
        f"PH2-BIC{tag}": (np.where(bicB, xB, xA), np.where(bicB, lB, lA)),
        f"PH2-A only{tag}": (xA, lA),
        f"PH2-B only{tag}": (xB, lB),
        f"PH2-avg{tag}": (wB * xB + (1 - wB) * xA, wB * lB + (1 - wB) * lA),
    }


est = {"Hill": (xi_of(df.aH), lq_of(df.qH)), "GPD": (xi_of(df.aG), lq_of(df.qG))}
est.update(ph2_variants("", ""))
if SHIFT:
    est["Hill (shift)"] = (xi_of(df.aH_s), lq_of(df.qH_s))
    est.update(ph2_variants("_s", " (shift)"))

L = pd.concat([pd.DataFrame(dict(dist=df.dist, seed=df.seed, k=df.k, method=m,
                                 xi=x, err_xi=x - XI, err_lq=l))
               for m, (x, l) in est.items()])
L["capped_q"] = L.err_lq.abs() >= CLIP
g = L.groupby(["dist", "method", "k"])
S = pd.DataFrame(dict(
    bias_xi=g.err_xi.mean(), rmse_xi=g.err_xi.apply(lambda e: np.sqrt(np.mean(e ** 2))),
    mad_xi=g.err_xi.apply(lambda e: np.median(np.abs(e))),
    bias_lq=g.err_lq.mean(), rmse_lq=g.err_lq.apply(lambda e: np.sqrt(np.mean(e ** 2))),
    capped_q=g.capped_q.mean())).reset_index()
S.to_csv(out("summary_by_k.csv"), index=False)

title = {d: f"{d}  (ρ = {DISTS[d][1]:g}{', in PH$_2$ class' if DISTS[d][2] else ''})" for d in order}
KGRID = sorted(df.k.unique())
KWIN = KGRID[1:]  # drop the smallest k from window summaries


# ---- figures (validated categorical palette; dash + marker as secondary encoding) -----
def panel(plot, metric, ylabel, fname, ref=None, ylim=None):
    nd = len(order)
    ncol = 3 if nd <= 6 else 4
    nrow = int(np.ceil(nd / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.75 * nrow), sharex=True,
                            squeeze=False)
    for ax, d in zip(axs.flat, order):
        for m, (c, ls, mk, lab) in plot.items():
            s = S[(S.dist == d) & (S.method == m)].sort_values("k")
            ax.plot(s.k, s[metric], ls, marker=mk, ms=5, lw=2, color=c, label=lab)
        if ref is not None:
            ax.axhline(ref, color="#888", lw=0.8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xscale("log"); ax.set_title(title[d], fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in list(axs.flat)[nd:]:
        ax.set_visible(False)
    for j in range(ncol):
        col = [axs[i, j] for i in range(nrow) if i * ncol + j < nd]
        col[-1].set_xlabel(f"k = number of exceedances (n = {N})")
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(plot), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out(fname), dpi=150); plt.close(fig)


BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
MAIN = {"Hill": (BLUE, "--", "s", "Hill"), "GPD": (ORANGE, "-.", "^", "GPD"),
        "PH2-B only": (AQUA, "-", "o", "PH$_2$ Exp+Erlang ($\\rho=0$ sub-model)"),
        "PH2-AIC": (VIOLET, ":", "D", "PH$_2$ AIC-selected")}
qlab = f"RMSE of log($\\hat Q$/Q),  Q = VaR$_{{{P:g}}}$"
panel(MAIN, "rmse_xi", "RMSE of $\\hat\\xi = 1/\\hat\\alpha$", "fig_rmse_xi.png", ylim=(0, 0.6))
panel(MAIN, "bias_xi", "Bias of $\\hat\\xi$", "fig_bias_xi.png", ref=0, ylim=(-0.3, 0.4))
panel(MAIN, "rmse_lq", qlab, "fig_rmse_q.png", ylim=(0, 1.5))
if SHIFT:
    SH = {"Hill": (BLUE, "--", "s", "Hill"),
          "Hill (shift)": (BLUE, ":", "x", "Hill, min-shift"),
          "GPD": (ORANGE, "-.", "^", "GPD (shift-invariant)"),
          "PH2-B only": (AQUA, "-", "o", "PH$_2$ Exp+Erlang"),
          "PH2-B only (shift)": (AQUA, ":", "x", "PH$_2$ Exp+Erlang, min-shift")}
    panel(SH, "rmse_xi", "RMSE of $\\hat\\xi$", "fig_shift_rmse_xi.png", ylim=(0, 0.6))
    panel(SH, "rmse_lq", qlab, "fig_shift_rmse_q.png", ylim=(0, 1.5))

# ---- summary table ------------------------------------------------------------------
K10, KMID, KMAX = KGRID[4], KGRID[6], KGRID[-1]  # 10 %, 20 %, 50 % of n
rows = []
for d in order:
    for m in est:
        s = S[(S.dist == d) & (S.method == m) & S.k.isin(KWIN)].set_index("k")
        w = L[(L.dist == d) & (L.method == m) & L.k.isin(KWIN)]
        rows.append({"dist": d, "method": m,
                     "xi_best": s.rmse_xi.min(), "k_best": s.rmse_xi.idxmin(),
                     "xi_worst": s.rmse_xi.max(), f"xi_k{K10}": s.rmse_xi.loc[K10],
                     f"xi_k{KMAX}": s.rmse_xi.loc[KMAX],
                     "q_best": s.rmse_lq.min(), "q_worst": s.rmse_lq.max(),
                     f"q_k{K10}": s.rmse_lq.loc[K10], f"q_k{KMAX}": s.rmse_lq.loc[KMAX],
                     "sens": w.groupby("seed").xi.std().median(),
                     "capped": s.capped_q.mean()})
T = pd.DataFrame(rows)
T.to_csv(out("summary_table.csv"), index=False)
pd.set_option("display.width", 250)
print(f"n = {N}, R = {df.seed.nunique() // len(order)} per DGP, VaR level p = {P:g}")
print(f"RMSE of xi-hat and of log VaR (k window {KWIN[0]}..{KWIN[-1]}); "
      "sens = median within-sample SD of xi-hat over k")
print(T.round(3).to_string(index=False))

# ---- second-order identification ------------------------------------------------------
for sfx in [""] + (["_s"] if SHIFT else []):
    for crit in ("AIC", "BIC"):
        fam = df[df.k.isin(KWIN)].groupby(["dist", "k"])[f"sel{crit}_B{sfx}"].mean().unstack()
        fam = fam.loc[order]
        fam.to_csv(out(f"ph2_share_rho0_{crit}{sfx}.csv"))
        print(f"\nShare of PH2{' (shift)' if sfx else ''} fits selecting rho = 0 "
              f"(Exp+Erlang) by {crit}:")
        print(fam.round(2).to_string())

# ---- does picking the 'wrong' sub-family hurt? (PH2-AIC, unshifted) -----------------------
KC = [int(K10), int(KMID), int(KMAX)]
print(f"\nPH2-AIC: RMSE of xi-hat by selected sub-family, k in {KC}")
sub = df[df.k.isin(KC)].copy()
sub["xi_sel"] = np.where(sub.selAIC_B, xi_of(sub.aB), xi_of(sub.aA))
sub["picked"] = np.where(sub.selAIC_B, "B (rho=0)", "A (rho<0)")
c = sub.groupby(["dist", "k", "picked"]).agg(
    n=("xi_sel", "size"), rmse=("xi_sel", lambda x: np.sqrt(np.mean((x - XI) ** 2))))
print(c.unstack("picked").round(3).loc[order].to_string())

ra = df[(~df.selAIC_B) & df.k.isin([KMID, KMAX])].groupby(["dist", "k"]).rhoA.median().unstack()
print("\nMedian rho-hat when sub-family A is selected:")
print(ra.reindex(order).round(2).to_string())
