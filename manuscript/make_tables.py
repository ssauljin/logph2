"""Generate the LaTeX tables of the manuscript from the simulation outputs in ../sim.

usage (from manuscript/):  python3 make_tables.py
Writes tables/*.tex. Every number in the simulation tables comes from these CSV files.
"""
import os, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "sim")
OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)

DGP_LABEL = {
    "PH2-Erlang": r"PH$_2$-Erlang", "Loggamma": "Loggamma", "Burr": "Burr",
    "Two-Pareto": "Two-Pareto", "GPD": "GPD", "GPD-loc5": r"GPD, $\mu=5$", "Frechet": r"Fr\'echet",
}
RHO = {"PH2-Erlang": "0", "Loggamma": "0", "Burr": "-0.5", "Two-Pareto": "-0.5",
       "GPD": "-1", "GPD-loc5": "-1", "Frechet": "-2"}


def fmt(v, bold=False):
    s = f"{v:.3f}"
    return rf"\textbf{{{s}}}" if bold else s


def rmse_table(tab, methods, labels, stat_cols, stat_labels, caption, label, fname, note=""):
    """Rows = DGPs; for each statistic, one column per method; row minimum in bold."""
    dists = [d for d in DGP_LABEL if d in set(tab.dist)]
    ncol = len(methods) * len(stat_cols)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{ll" + "r" * ncol + "}", r"\toprule"]
    head1 = r" & & " + " & ".join(
        rf"\multicolumn{{{len(methods)}}}{{c}}{{{sl}}}" for sl in stat_labels) + r" \\"
    rules = " ".join(rf"\cmidrule(lr){{{3 + i * len(methods)}-{2 + (i + 1) * len(methods)}}}"
                     for i in range(len(stat_cols)))
    head2 = r"DGP & $\rho$ & " + " & ".join(labels * len(stat_cols)) + r" \\"
    lines += [head1, rules, head2, r"\midrule"]
    for d in dists:
        row = [DGP_LABEL[d], f"${RHO[d]}$"]
        for sc in stat_cols:
            vals = [tab[(tab.dist == d) & (tab.method == m)][sc].iloc[0] for m in methods]
            vmin = min(vals)
            row += [fmt(v, round(v, 3) == round(vmin, 3)) for v in vals]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [r"\par\smallskip", rf"\parbox{{\linewidth}}{{\footnotesize {note}}}"]
    lines += [r"\end{table}"]
    open(os.path.join(OUT, fname), "w").write("\n".join(lines) + "\n")


T2 = pd.read_csv(os.path.join(SIM, "summary_table.csv"))
T10 = pd.read_csv(os.path.join(SIM, "n10000", "summary_table.csv"))

M4 = ["Hill", "GPD", "PH2-B only", "PH2-AIC"]
L4 = ["Hill", "GPD", "E+E", "AIC"]
note = (r"E+E: PH$_2$ Exp+Erlang ($\rho=0$) sub-model; AIC: PH$_2$ with the sub-family chosen by AIC. "
        r"``Best'' is the smallest RMSE over the threshold grid (oracle threshold); ``worst'' is the "
        r"largest RMSE over the grid, a measure of sensitivity to the threshold choice. "
        r"The smallest value in each row and block is in bold.")
rmse_table(T2, M4, L4, ["xi_best", "xi_worst"], ["Best $k$", "Worst $k$"],
           r"RMSE of $\hat\xi$ over the threshold grid $k\in\{50,\dots,1000\}$, $n=2000$, $R=500$.",
           "tab:xi2000", "tab_xi_n2000.tex", note)
rmse_table(T2, M4, L4, ["q_best", "q_worst"], ["Best $k$", "Worst $k$"],
           r"RMSE of $\log(\widehat{\mathrm{VaR}}_{0.9995}/\mathrm{VaR}_{0.9995})$ over "
           r"$k\in\{50,\dots,1000\}$, $n=2000$, $R=500$.",
           "tab:q2000", "tab_q_n2000.tex",
           r"Log-ratio errors are truncated at $\pm 3$ before averaging. " + note.split(". ", 1)[1])

M5 = ["Hill", "GPD", "PH2-B only", "PH2-B only (shift)", "PH2-AIC"]
L5 = ["Hill", "GPD", "E+E", "E+E$_s$", "AIC"]
rmse_table(T10, M5, L5, ["xi_best", "xi_worst"], ["Best $k$", "Worst $k$"],
           r"RMSE of $\hat\xi$ over $k\in\{250,\dots,5000\}$, $n=10\,000$, $R=500$.",
           "tab:xi10000", "tab_xi_n10000.tex",
           r"E+E$_s$: Exp+Erlang sub-model fitted to the min-shifted log-excesses "
           r"$\log\{(Y-m)/(u-m)\}$, $m=\min_i Y_i$. For Loggamma and Two-Pareto the small best-$k$ "
           r"value of E+E$_s$ occurs where its bias changes sign and should not be read as a uniform "
           r"advantage. " + note)
rmse_table(T10, M5, L5, ["q_best", "q_worst"], ["Best $k$", "Worst $k$"],
           r"RMSE of $\log(\widehat{\mathrm{VaR}}_{0.9999}/\mathrm{VaR}_{0.9999})$ over "
           r"$k\in\{250,\dots,5000\}$, $n=10\,000$, $R=500$.",
           "tab:q10000", "tab_q_n10000.tex", r"Log-ratio errors are truncated at $\pm 3$. See Table~\ref{tab:xi10000}.")


# ---- selection frequency of the rho = 0 sub-family ---------------------------------
def share_table():
    s2 = pd.read_csv(os.path.join(SIM, "ph2_share_rho0_AIC.csv"), index_col=0)
    b2 = pd.read_csv(os.path.join(SIM, "ph2_share_rho0_BIC.csv"), index_col=0)
    s10 = pd.read_csv(os.path.join(SIM, "n10000", "ph2_share_rho0_AIC.csv"), index_col=0)
    b10 = pd.read_csv(os.path.join(SIM, "n10000", "ph2_share_rho0_BIC.csv"), index_col=0)
    k2, k10 = ["100", "400", "1000"], ["500", "2000", "5000"]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Proportion of samples in which the $\rho=0$ (Exp+Erlang) sub-family is "
             r"selected over the two-exponential sub-family ($\rho<0$).}",
             r"\label{tab:share}", r"\begin{tabular}{llrrrrrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{6}{c}{$n=2000$, $k=$} & \multicolumn{6}{c}{$n=10\,000$, $k=$} \\",
             r"\cmidrule(lr){3-8} \cmidrule(lr){9-14}",
             r" & & \multicolumn{3}{c}{AIC} & \multicolumn{3}{c}{BIC} & \multicolumn{3}{c}{AIC} "
             r"& \multicolumn{3}{c}{BIC} \\",
             r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(lr){12-14}",
             r"DGP & $\rho$ & " + " & ".join(k2 + k2 + k10 + k10) + r" \\", r"\midrule"]
    for d in DGP_LABEL:
        row = [DGP_LABEL[d], f"${RHO[d]}$"]
        for tab, ks in ((s2, k2), (b2, k2), (s10, k10), (b10, k10)):
            row += [f"{tab.loc[d, k]:.2f}" if d in tab.index else "--" for k in ks]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\parbox{\linewidth}{\footnotesize The GPD with location $\mu=5$ was included only "
              r"in the $n=10\,000$ study. Fits are on the unshifted log-excesses.}",
              r"\end{table}"]
    open(os.path.join(OUT, "tab_share.tex"), "w").write("\n".join(lines) + "\n")


share_table()


# ---- conditional accuracy and rho-hat (parsed from results_tables.txt) -------------------
def cond_table():
    rows = []
    for n, path, ks in ((2000, os.path.join(SIM, "results_tables.txt"), ("400", "1000")),
                        (10000, os.path.join(SIM, "n10000", "results_tables.txt"), ("2000", "5000"))):
        txt = open(path).read()
        blk = txt.split("PH2-AIC: RMSE of xi-hat by selected sub-family")[1].split("Median rho-hat")[0]
        cur = None
        for line in blk.splitlines()[4:]:
            parts = line.split()
            if not parts:
                continue
            if not re.match(r"^\d", parts[0]):
                cur, parts = parts[0], parts[1:]
            k, nA, nB, rA, rB = parts
            if k in ks:
                rows.append((n, cur, k, nA, nB, rA, rB))
        rh = txt.split("Median rho-hat when sub-family A is selected:")[1].strip().splitlines()
        rhos = {l.split()[0]: l.split()[1:] for l in rh[2:]}
        rows = [r + (rhos.get(r[1], ["nan", "nan"])[ks.index(r[2])],) if r[0] == n else r
                for r in rows]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{RMSE of $\hat\xi$ for the AIC-selected PH$_2$ fit, split by the selected "
             r"sub-family, and median $\hat\rho=-\hat\tau$ among fits selecting the two-exponential "
             r"sub-family.}", r"\label{tab:cond}",
             r"\begin{tabular}{rlrrrrrr}", r"\toprule",
             r"$n$ & DGP ($\rho$) & $k$ & \#$\rho<0$ & \#$\rho=0$ & RMSE, $\rho<0$ "
             r"& RMSE, $\rho=0$ & median $\hat\rho$ \\", r"\midrule"]
    last = None
    for (n, d, k, nA, nB, rA, rB, rho) in rows:
        f = lambda v: "--" if v == "NaN" or v == "nan" else v
        nn = f"{n}" if n != last else ""
        if last is not None and n != last:
            lines.append(r"\midrule")
        last = n
        lines.append(" & ".join([nn, f"{DGP_LABEL[d]} (${RHO[d]}$)", k,
                                 f(nA).replace(".0", ""), f(nB).replace(".0", ""),
                                 f(rA), f(rB), f(rho)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_cond.tex"), "w").write("\n".join(lines) + "\n")


cond_table()


# ---- variance inflation of the Exp+Erlang MLE (closed-form Fisher information) ----------
def infl_table():
    import sys
    sys.path.insert(0, SIM)
    from fisher_B import avar_xi
    ps = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    vals = [avar_xi(1.0, p) for p in ps]  # avar/xi^2 is free of mu; mu = 1 gives xi = 1
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Asymptotic variance of the Exp+Erlang MLE $\hat\xi$, relative to $\xi^2$ "
             r"(the asymptotic variance of the Hill estimator under an exact Pareto tail), "
             r"as a function of the Erlang weight $p$.}", r"\label{tab:infl}",
             r"\begin{tabular}{l" + "r" * len(ps) + "}", r"\toprule",
             r"$p$ & " + " & ".join(f"{p:.2f}" for p in ps) + r" \\", r"\midrule",
             r"$\sigma^2_\xi(p)/\xi^2$ & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_infl.tex"), "w").write("\n".join(lines) + "\n")


infl_table()


# ---- coverage table (parsed from coverage_n2000.txt, if present) -------------------------
def cov_table():
    path = os.path.join(SIM, "coverage_n2000.txt")
    if not os.path.exists(path):
        return
    rx = re.compile(r"^(\S+)\s+(\d+)\s+([\d.]+)\s+\[\s*([\d.]+)\]\s+([\d.]+)\s+\[\s*([\d.]+)\]\s+"
                    r"([\d.]+)\s+\[\s*([\d.]+)\]\s+([\d.]+)")
    rows = [m.groups() for m in map(rx.match, open(path)) if m]
    if not rows:
        return
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Empirical coverage (mean width) of nominal 95\% intervals for $\xi=0.5$, "
             r"$n=2000$, $R=500$.}", r"\label{tab:cov}",
             r"\begin{tabular}{lrlllr}", r"\toprule",
             r"DGP & $k$ & Wald, E+E & Profile, E+E & Hill & $\Pr(\hat p<0.01)$ \\", r"\midrule"]
    last = None
    for d, k, cW, wW, cP, wP, cH, wH, pb in rows:
        if last is not None and d != last:
            lines.append(r"\midrule")
        lines.append(" & ".join([DGP_LABEL[d] if d != last else "", k,
                                 f"{float(cW):.3f} ({float(wW):.3f})",
                                 f"{float(cP):.3f} ({float(wP):.3f})",
                                 f"{float(cH):.3f} ({float(wH):.3f})", f"{float(pb):.2f}"]) + r" \\")
        last = d
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\parbox{\linewidth}{\footnotesize Wald intervals whose width exceeds 10, or whose estimated "
              r"information is numerically singular (near $\hat p=0$), are treated as uninformative: width 10, "
              r"counted as covering. Last column: proportion of fits with $\hat p<0.01$.}",
              r"\end{table}"]
    open(os.path.join(OUT, "tab_cov.tex"), "w").write("\n".join(lines) + "\n")


cov_table()
print("tables written:", sorted(os.listdir(OUT)))
