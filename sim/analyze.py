import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ph2sim import DISTS, ALPHA, true_quantile

P = 0.9995
XI = 1 / ALPHA
CLIP = 3.0     # cap |log(Qhat/Q)|; share of capped values is reported
XICAP = 5.0    # cap xi-hat (alpha-hat -> 0 explosions)
import os
df = pd.read_csv("mc_results.csv" if os.path.exists("mc_results.csv") else "mc_results.csv.gz")
df["qt"] = df["dist"].map({d: true_quantile(d, P) for d in DISTS})

# ---- PH2 variants built from the two stored sub-family fits -------------------------
aicA, aicB = 2 * df.nllA + 6, 2 * df.nllB + 4
bicA, bicB = 2 * df.nllA + 3 * np.log(df.k), 2 * df.nllB + 2 * np.log(df.k)
df["selAIC_B"] = aicB <= aicA
df["selBIC_B"] = bicB <= bicA


def xi_of(a):
    a = np.asarray(a, float)
    return np.clip(np.where(np.isfinite(a) & (a > 0), 1 / a, XICAP), None, XICAP)


def lq_of(q):
    q = np.asarray(q, float)
    out = np.log(np.clip(q, 1e-300, None) / df.qt.values)
    out[~np.isfinite(out)] = CLIP
    return np.clip(out, -CLIP, CLIP)


est = {
    "Hill": (xi_of(df.aH), lq_of(df.qH)),
    "GPD": (xi_of(df.aG), lq_of(df.qG)),
    "PH2-AIC": (np.where(df.selAIC_B, xi_of(df.aB), xi_of(df.aA)),
                np.where(df.selAIC_B, lq_of(df.qB), lq_of(df.qA))),
    "PH2-BIC": (np.where(df.selBIC_B, xi_of(df.aB), xi_of(df.aA)),
                np.where(df.selBIC_B, lq_of(df.qB), lq_of(df.qA))),
    "PH2-A only": (xi_of(df.aA), lq_of(df.qA)),
    "PH2-B only": (xi_of(df.aB), lq_of(df.qB)),
}
wB = 1 / (1 + np.exp(-(aicA - aicB) / 2))  # Akaike weight of sub-family B
est["PH2-avg"] = (wB * xi_of(df.aB) + (1 - wB) * xi_of(df.aA),
                  wB * lq_of(df.qB) + (1 - wB) * lq_of(df.qA))

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
S.to_csv("summary_by_k.csv", index=False)

order = list(DISTS)
title = {d: f"{d}  (ρ = {DISTS[d][1]:g}{', in PH$_2$ class' if DISTS[d][2] else ''})" for d in order}

# ---- figures: Hill vs GPD vs PH2-AIC (validated 3-colour palette + dash/marker) ------
PLOT = {"Hill": ("#2a78d6", "--", "s"), "GPD": ("#eb6834", "-.", "^"),
        "PH2-B only": ("#1baf7a", "-", "o"), "PH2-AIC": ("#6250d6", ":", "D")}
LABEL = {"Hill": "Hill", "GPD": "GPD", "PH2-B only": "PH$_2$ Exp+Erlang ($\\rho=0$ sub-model)",
         "PH2-AIC": "PH$_2$ AIC-selected"}


def panel(metric, ylabel, fname, ref=None, ylim=None):
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for ax, d in zip(axs.flat, order):
        for m, (c, ls, mk) in PLOT.items():
            s = S[(S.dist == d) & (S.method == m)].sort_values("k")
            ax.plot(s.k, s[metric], ls, marker=mk, ms=5, lw=2, color=c, label=LABEL[m])
        if ref is not None:
            ax.axhline(ref, color="#888", lw=0.8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xscale("log"); ax.set_title(title[d], fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axs[1]:
        ax.set_xlabel("k = number of exceedances (n = 2000)")
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(fname, dpi=150); plt.close(fig)


panel("rmse_xi", "RMSE of $\\hat\\xi = 1/\\hat\\alpha$", "fig_rmse_xi.png", ylim=(0, 0.6))
panel("bias_xi", "Bias of $\\hat\\xi$", "fig_bias_xi.png", ref=0, ylim=(-0.3, 0.4))
panel("rmse_lq", "RMSE of log($\\hat Q$/Q),  Q = VaR$_{0.9995}$", "fig_rmse_q.png", ylim=(0, 1.5))

# ---- summary table ----------------------------------------------------------------
KWIN = [50, 100, 150, 200, 300, 400, 600, 800, 1000]
rows = []
for d in order:
    for m in est:
        s = S[(S.dist == d) & (S.method == m) & S.k.isin(KWIN)].set_index("k")
        w = L[(L.dist == d) & (L.method == m) & L.k.isin(KWIN)]
        rows.append(dict(dist=d, method=m,
                         xi_best=s.rmse_xi.min(), k_best=s.rmse_xi.idxmin(),
                         xi_worst=s.rmse_xi.max(), xi_k200=s.rmse_xi.loc[200],
                         xi_k1000=s.rmse_xi.loc[1000],
                         q_best=s.rmse_lq.min(), q_worst=s.rmse_lq.max(),
                         q_k200=s.rmse_lq.loc[200], q_k1000=s.rmse_lq.loc[1000],
                         sens=w.groupby("seed").xi.std().median(),
                         capped=s.capped_q.mean()))
T = pd.DataFrame(rows)
T.to_csv("summary_table.csv", index=False)
pd.set_option("display.width", 250)
print("RMSE of xi-hat and of log VaR (k window 50..1000); sens = median within-sample SD of xi-hat over k")
print(T.round(3).to_string(index=False))

# ---- second-order identification ---------------------------------------------------
for crit in ("AIC", "BIC"):
    fam = df[df.k.isin(KWIN)].groupby(["dist", "k"])[f"sel{crit}_B"].mean().unstack().loc[order]
    fam.to_csv(f"ph2_share_rho0_{crit}.csv")
    print(f"\nShare of PH2 fits selecting rho = 0 (Exp+Erlang) by {crit}:")
    print(fam.round(2).to_string())

# ---- does picking the 'wrong' sub-family hurt? (conditional accuracy, PH2-AIC) -------
print("\nPH2-AIC: RMSE of xi-hat by selected sub-family, k in {200, 400, 1000}")
sub = df[df.k.isin([200, 400, 1000])].copy()
sub["xi_sel"] = np.where(sub.selAIC_B, xi_of(sub.aB), xi_of(sub.aA))
sub["picked"] = np.where(sub.selAIC_B, "B (rho=0)", "A (rho<0)")
c = sub.groupby(["dist", "k", "picked"]).agg(
    n=("xi_sel", "size"), rmse=("xi_sel", lambda x: np.sqrt(np.mean((x - XI) ** 2))))
print(c.unstack("picked").round(3).loc[order].to_string())

# rho-hat from family A when selected
ra = df[(~df.selAIC_B) & df.k.isin([400, 1000])].groupby(["dist", "k"]).rhoA.median().unstack()
print("\nMedian rho-hat when sub-family A is selected:")
print(ra.reindex(order).round(2).to_string())
