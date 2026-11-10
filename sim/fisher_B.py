"""
Fisher information for the Exp+Erlang(2) PH2 sub-model of the log-excess
    f(x; mu, p) = (1-p) mu e^{-mu x} + p mu^2 x e^{-mu x},   0 <= p <= 1,
and the resulting Wald interval for the tail index xi = 1/mu.

With Z = mu X, a = (1-p)/p and J0 = e^a E1(a):
    I_mumu = (1 + p(1 - a + a^2 J0)) / mu^2
    I_mup  = -(1 - a J0) / (mu p)
    I_pp   = (J0 - p) / p^3

usage:
  python3 fisher_B.py check      # verify the closed form by Monte Carlo score variance
  python3 fisher_B.py table      # variance inflation of xi-hat relative to Hill
  python3 fisher_B.py coverage   # coverage of Wald CIs for xi (small simulation)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import sys
import numpy as np
from scipy import special, optimize, stats


def _scaled_e1(a):
    """e^a E1(a); direct evaluation up to a = 700, asymptotic series beyond (exp overflow)."""
    if a > 700:
        return (1 - 1 / a + 2 / a ** 2 - 6 / a ** 3 + 24 / a ** 4) / a
    return special.exp1(a) * np.exp(a)


def fisher_B(mu, p):
    """Per-observation expected information matrix for (mu, p), 0 < p <= 1."""
    a = (1 - p) / p
    J0 = _scaled_e1(a)
    if a == 0:  # p = 1: pure Erlang(2); limits a J0 -> 0, a^2 J0 -> 0
        Imm, Imp, Ipp = (1 + 1.0) / mu ** 2, -1.0 / mu, np.inf
        return np.array([[Imm, Imp], [Imp, Ipp]])
    Imm = (1 + p * (1 - a + a * a * J0)) / mu ** 2
    Imp = -(1 - a * J0) / (mu * p)
    Ipp = (J0 - p) / p ** 3
    return np.array([[Imm, Imp], [Imp, Ipp]])


def avar_xi(mu, p):
    """Asymptotic variance of sqrt(k)(xi_hat - xi), xi = 1/mu (delta method)."""
    Iinv = np.linalg.inv(fisher_B(mu, p))
    return Iinv[0, 0] / mu ** 4


def rB(n, rng, mu, p):
    return rng.gamma(np.where(rng.random(n) < p, 2, 1), 1 / mu)


def score(x, mu, p):
    D = 1 - p + p * mu * x
    return np.stack([1 / mu - x + p * x / D, (mu * x - 1) / D])


def check():
    rng = np.random.default_rng(1)
    for mu, p in [(2.0, 0.2), (2.0, 0.5), (1.0, 0.8), (3.0, 0.05)]:
        x = rB(2_000_000, rng, mu, p)
        s = score(x, mu, p)
        print(f"mu={mu}, p={p}\n closed form:\n{fisher_B(mu, p).round(4)}\n"
              f" MC score covariance:\n{np.cov(s).round(4)}\n mean score {s.mean(1).round(4)}")


def table():
    print("Asymptotic variance of xi-hat (Exp+Erlang MLE) relative to Hill (xi^2), mu = 1/xi:")
    print("  p    ratio")
    for p in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        print(f" {p:4.2f}  {avar_xi(2.0, p) * 2.0 ** 2:6.2f}")  # avar / xi^2 is free of mu


def nll(par, x):
    a, c = par
    mu, p = np.exp(a), special.expit(c)
    return -(len(x) * a - mu * x.sum() + np.log1p(p * (mu * x - 1)).sum())


def fitB(x):
    h = 1 / x.mean()
    best = None
    for ft in (0.8, 1.2, 1.8):
        for c0 in (-2.0, 0.0, 2.0):
            r = optimize.minimize(nll, (np.log(h * ft), c0), args=(x,), method="L-BFGS-B",
                                  bounds=[(-8, 6), (-30, 30)])
            if best is None or r.fun < best.fun:
                best = r
    return np.exp(best.x[0]), special.expit(best.x[1])


def profile_ci_xi(x, mu_hat, level=0.95):
    """Profile-likelihood CI for xi = 1/mu, maximising over p in [0, 1] (boundary allowed)."""
    def lp(mu):  # profile log-likelihood
        f = lambda p: -(len(x) * np.log(mu) - mu * x.sum() + np.log1p(p * (mu * x - 1)).sum())
        r = optimize.minimize_scalar(f, bounds=(0.0, 1.0), method="bounded",
                                     options=dict(xatol=1e-7))
        return -r.fun
    lmax = lp(mu_hat)
    crit = stats.chi2.ppf(level, 1) / 2
    g = lambda mu: lmax - lp(mu) - crit
    lo = mu_hat
    for _ in range(60):
        lo *= 0.9
        if g(lo) > 0:
            break
    hi = mu_hat
    for _ in range(60):
        hi *= 1.1
        if g(hi) > 0:
            break
    mlo = optimize.brentq(g, lo, mu_hat * (1 - 1e-9))
    mhi = optimize.brentq(g, mu_hat * (1 + 1e-9), hi)
    return 1 / mhi, 1 / mlo


def coverage(R=500):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ph2sim import DISTS
    rng = np.random.default_rng(2024)
    z = stats.norm.ppf(0.975)
    print(f"Coverage of nominal 95% CIs for xi = 0.5 (n = 2000, R = {R}); "
          "mean width in brackets; pbound = share of fits with p-hat < 0.01")
    print(f"{'DGP':12s} {'k':>5s} {'Wald (E+E)':>16s} {'Profile (E+E)':>16s} {'Hill':>16s} {'pbound':>7s}")
    for name in ("PH2-Erlang", "Loggamma", "Two-Pareto", "GPD"):
        for k in (200, 500, 1000):
            cW = cP = cH = 0
            wW = wP = wH = 0.0
            nb = 0
            for r in range(R):
                y = np.sort(DISTS[name][0](2000, rng))
                u = y[-k - 1]
                x = np.log(y[-k:] / u)
                mu, p = fitB(x)
                nb += p < 0.01
                av = avar_xi(mu, min(max(p, 1e-6), 1 - 1e-9))
                if not np.isfinite(av) or av <= 0 or 2 * z * np.sqrt(av / k) > 10:
                    cW += 1          # numerically singular information near p = 0:
                    wW += 10.0       # treated as an uninformative interval of width 10
                else:
                    se = np.sqrt(av / k)
                    cW += abs(1 / mu - 0.5) <= z * se
                    wW += 2 * z * se
                a, b = profile_ci_xi(x, mu)
                cP += a <= 0.5 <= b
                wP += b - a
                xh = x.mean()
                seH = xh / np.sqrt(k)
                cH += abs(xh - 0.5) <= z * seH
                wH += 2 * z * seH
            print(f"{name:12s} {k:5d} {cW / R:6.3f} [{wW / R:6.3f}] {cP / R:6.3f} [{wP / R:6.3f}] "
                  f"{cH / R:6.3f} [{wH / R:6.3f}] {nb / R:7.3f}", flush=True)


if __name__ == "__main__":
    {"check": check, "table": table, "coverage": coverage}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
