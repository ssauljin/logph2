"""
Monte Carlo study: threshold-robustness of PH2 (log-excess) tail fitting vs Hill and GPD.

Idea
----
All three estimators use the same k exceedances over u = Y_(n-k):
  * Hill  : exponential MLE on log(Y/u)                 -> first order only (rho ignored)
  * GPD   : GPD MLE on Y - u                             -> first order only (rho = -1 built in)
  * PH2   : PH2 MLE on log(Y/u), two sub-families         -> second-order aware
        (A) two-exponential (generalised hyperexponential, w may be < 0):
            S(x) = (1-w) e^{-theta x} + w e^{-(theta+tau) x},   alpha = theta, rho = -tau
        (B) exponential + Erlang(2), common rate:
            S(x) = e^{-mu x} (1 + p mu x),                     alpha = mu,    rho = 0
        Sub-family chosen by AIC.
Because the second-order term is part of the PH2 model, its bias should not grow with k
the way Hill/GPD bias does, so it can use a much lower threshold (more data) => lower RMSE
and a flat, threshold-insensitive estimate. That flatness is the "robustness" claim.

rho below uses the tail-function convention of the draft (eq. 2nd.order.cond):
  (1-F(tx))/(1-F(t)) - x^{-alpha} ~ a(t) x^{-alpha}(x^rho - 1)/rho,  a(t) in RV_rho.
(The U-quantile convention used by much of the EVT literature is rho/alpha.)
"""
import numpy as np
from scipy import optimize, special, stats

# --------------------------------------------------------------------------------------
# Data-generating processes (all with alpha = 2, i.e. xi = 0.5)
# --------------------------------------------------------------------------------------
ALPHA = 2.0


def r_ph2_erlang(n, rng, mu=2.0, p=0.5):
    # log Y ~ (1-p) Exp(mu) + p Erlang(2, mu)  -> Pareto/log-gamma mixture, rho = 0 (in class)
    shape = np.where(rng.random(n) < p, 2, 1)
    return np.exp(rng.gamma(shape, 1 / mu))


def s_ph2_erlang(y, mu=2.0, p=0.5):
    x = np.log(y)
    return np.exp(-mu * x) * (1 + p * mu * x)


def r_two_pareto(n, rng, theta=2.0, tau=0.5, w=0.7):
    # log Y ~ (1-w) Exp(theta) + w Exp(theta+tau)  -> two-Pareto mixture, rho = -tau (in class)
    rate = np.where(rng.random(n) < w, theta + tau, theta)
    return np.exp(rng.exponential(1 / rate))


def s_two_pareto(y, theta=2.0, tau=0.5, w=0.7):
    return (1 - w) * y ** (-theta) + w * y ** (-(theta + tau))


def r_loggamma(n, rng, a=1.5, rate=2.0):
    # log Y ~ Gamma(1.5, rate 2): tail ~ (log y)^{0.5} y^{-2}, rho = 0 (NOT in PH2 class)
    return np.exp(rng.gamma(a, 1 / rate, n))


def r_burr(n, rng, c=0.5, k=4.0):
    # S(y) = (1 + y^c)^{-k}: alpha = c k = 2, rho = -c = -0.5 (not in class; pairs with Two-Pareto)
    u = rng.random(n)
    return (u ** (-1 / k) - 1) ** (1 / c)


def r_gpd(n, rng, xi=0.5, sigma=1.0):
    # GPD: alpha = 2, rho = -1
    u = rng.random(n)
    return sigma / xi * (u ** (-xi) - 1)


def r_gpd_loc5(n, rng, xi=0.5, sigma=1.0, mu=5.0):
    # GPD with location 5 (as in JBF Fig. 3): alpha = 2, rho = -1; support starts at 5,
    # the case the JBF min-shift log(Y - min Y) is designed for
    return mu + r_gpd(n, rng, xi, sigma)


def r_frechet(n, rng, a=2.0):
    # Frechet: alpha = 2, rho = -2
    u = rng.random(n)
    return (-np.log(u)) ** (-1 / a)


def _quantile_from_survival(sfun, q, lo=1.0, hi=1e12):
    return optimize.brentq(lambda y: sfun(y) - q, lo, hi, xtol=1e-12, rtol=1e-12)


def true_quantile(name, p):
    q = 1 - p
    if name == "PH2-Erlang":
        return _quantile_from_survival(s_ph2_erlang, q)
    if name == "Two-Pareto":
        return _quantile_from_survival(s_two_pareto, q)
    if name == "Loggamma":
        return np.exp(stats.gamma.isf(q, 1.5, scale=1 / 2.0))
    if name == "Burr":
        return (q ** (-1 / 4.0) - 1) ** (1 / 0.5)
    if name == "GPD-loc5":
        return 5 + 1 / 0.5 * (q ** (-0.5) - 1)
    if name == "GPD":
        return 1 / 0.5 * (q ** (-0.5) - 1)
    if name == "Frechet":
        return (-np.log(p)) ** (-1 / 2.0)
    raise KeyError(name)


DISTS = {
    # name: (sampler, rho, in PH2 class?)
    "PH2-Erlang": (r_ph2_erlang, 0.0, True),
    "Loggamma": (r_loggamma, 0.0, False),
    "Burr": (r_burr, -0.5, False),
    "Two-Pareto": (r_two_pareto, -0.5, True),
    "GPD": (r_gpd, -1.0, False),
    "GPD-loc5": (r_gpd_loc5, -1.0, False),
    "Frechet": (r_frechet, -2.0, False),
}

# --------------------------------------------------------------------------------------
# Estimators. Each returns (alpha_hat, quantile_fn(level q of the EXCEEDANCE survival))
# --------------------------------------------------------------------------------------


def fit_hill(x):
    """x = log(Y/u) > 0."""
    a = 1 / x.mean()
    return a, (lambda qq: np.log(1 / qq) / a)  # log-excess quantile


def fit_gpd(z):
    """z = Y - u > 0. MLE for (xi, sigma)."""
    m, v = z.mean(), z.var()
    xi0 = np.clip(0.5 * (1 - m * m / v), -0.4, 0.9)
    s0 = max(m * (1 - xi0), 1e-8)

    def nll(par):
        xi, ls = par
        s = np.exp(ls)
        t = 1 + xi * z / s
        if np.any(t <= 0):
            return 1e100
        if abs(xi) < 1e-8:
            return len(z) * ls + z.sum() / s
        return len(z) * ls + (1 + 1 / xi) * np.log(t).sum()

    best = None
    for st in [(xi0, np.log(s0)), (0.5, np.log(m * 0.5)), (0.1, np.log(m * 0.9))]:
        r = optimize.minimize(nll, st, method="Nelder-Mead",
                              options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    xi, s = best.x[0], np.exp(best.x[1])
    alpha = 1 / xi if xi > 0 else np.inf

    def qfun(qq):  # quantile of exceedance Y - u
        if abs(xi) < 1e-8:
            return s * np.log(1 / qq)
        return s / xi * (qq ** (-xi) - 1)

    return alpha, xi, qfun


def _nll_A(par, x):
    """Two-exponential PH2 on log-excess. theta=e^a, tau=e^b, w in (-theta/tau, 1).
    Returns (nll, gradient) for L-BFGS-B."""
    a, b, c = par
    th, ta = np.exp(a), np.exp(b)
    lo = -th / ta
    s = special.expit(c)
    w = lo + (1 - lo) * s
    E = np.exp(-ta * x)
    inner = (1 - w) * th + w * (th + ta) * E
    if np.any(inner <= 0) or not np.isfinite(inner).all():
        return 1e100, np.zeros(3)
    nll = th * x.sum() - np.log(inner).sum()
    d_th = (1 - w) + w * E                       # d inner / d theta (w fixed)
    d_ta = w * E * (1 - (th + ta) * x)           # d inner / d tau   (w fixed)
    d_w = -th + (th + ta) * E                    # d inner / d w
    dw_dth, dw_dta, dw_dc = -(1 - s) / ta, (1 - s) * th / ta ** 2, (1 - lo) * s * (1 - s)
    g_th = x.sum() - ((d_th + d_w * dw_dth) / inner).sum()
    g_ta = -((d_ta + d_w * dw_dta) / inner).sum()
    g_c = -((d_w * dw_dc) / inner).sum()
    return nll, np.array([g_th * th, g_ta * ta, g_c])


def _nll_B(par, x):
    """Exponential + Erlang(2) PH2 on log-excess. mu=e^a, p=expit(c)."""
    a, c = par
    mu, p = np.exp(a), special.expit(c)
    g = p * (mu * x - 1)
    nll = -(len(x) * a - mu * x.sum() + np.log1p(g).sum())
    ga = -(len(x) - mu * x.sum() + mu * (p * x / (1 + g)).sum())
    gc = -((mu * x - 1) / (1 + g)).sum() * p * (1 - p)
    return nll, np.array([ga, gc])


def _sf_A(th, ta, w):
    return lambda t: (1 - w) * np.exp(-th * t) + w * np.exp(-(th + ta) * t)


def _sf_B(mu, p):
    return lambda t: np.exp(-mu * t) * (1 + p * mu * t)


def fit_ph2_both(x):
    """Fit both PH2 sub-families to x = log(Y/u) > 0.
    Returns dict A: (alpha, rho, nll, sf), B: (alpha, 0, nll, sf)."""
    h = 1 / x.mean()
    bestA = None
    for ft in (0.5, 0.8, 1.0):
        for rt in (0.25, 1.0, 4.0):
            for c0 in (-1.5, 1.5):
                st = (np.log(h * ft), np.log(h * rt), c0)
                r = optimize.minimize(_nll_A, st, args=(x,), jac=True, method="L-BFGS-B",
                                      bounds=[(-8, 6), (-8, 6), (-30, 30)],
                                      options=dict(ftol=1e-12, gtol=1e-8))
                if bestA is None or r.fun < bestA.fun:
                    bestA = r
    bestB = None
    for ft in (0.8, 1.2, 1.8):
        for c0 in (-2.0, 0.0, 2.0):
            r = optimize.minimize(_nll_B, (np.log(h * ft), c0), args=(x,), jac=True,
                                  method="L-BFGS-B", bounds=[(-8, 6), (-30, 30)],
                                  options=dict(ftol=1e-12, gtol=1e-8))
            if bestB is None or r.fun < bestB.fun:
                bestB = r
    a, b, c = bestA.x
    th, ta = np.exp(a), np.exp(b)
    lo = -th / ta
    w = lo + (1 - lo) * special.expit(c)
    mu, p = np.exp(bestB.x[0]), special.expit(bestB.x[1])
    return dict(A=(th, -ta, bestA.fun, _sf_A(th, ta, w)),
                B=(mu, 0.0, bestB.fun, _sf_B(mu, p)))


def fit_ph2(x):
    """AIC-selected PH2 (kept for backward compatibility)."""
    f = fit_ph2_both(x)
    if 2 * f["B"][2] + 4 <= 2 * f["A"][2] + 6:
        return f["B"][0], 0.0, "B", f["B"][3], None
    return f["A"][0], f["A"][1], "A", f["A"][3], None


def ph2_quantile(sf, qq):
    if qq >= 1:
        return 0.0
    hi = 1.0
    while sf(hi) > qq:
        hi *= 2
        if hi > 1e6:
            return np.inf
    return optimize.brentq(lambda t: sf(t) - qq, 0.0, hi, xtol=1e-12)


# --------------------------------------------------------------------------------------
# One replication
# --------------------------------------------------------------------------------------


def _log_scale_fits(x, u, qq, scale):
    """Hill + both PH2 sub-families on log-excess x; quantiles mapped back by `scale`."""
    aH = 1 / x.mean()
    qH = scale(qq ** (-1 / aH))
    f = fit_ph2_both(x)
    aA, rA, nA, sA = f["A"]
    aB, _, nB, sB = f["B"]
    qA = scale(np.exp(ph2_quantile(sA, qq)))
    qB = scale(np.exp(ph2_quantile(sB, qq)))
    return aH, qH, aA, qA, rA, nA, aB, qB, nB


def one_rep(args):
    """Unshifted log(Y/u) fits and JBF min-shifted log((Y-m)/(u-m)) fits, m = min(Y).
    GPD on Y - u is shift-invariant, so it is fitted once."""
    name, seed, n, kgrid, p = args
    rng = np.random.default_rng(seed)
    y = np.sort(DISTS[name][0](n, rng))
    m = y[0]
    out = []
    for k in kgrid:
        u = y[n - k - 1]
        ex = y[n - k:]
        qq = n * (1 - p) / k  # exceedance-survival level for target quantile
        aG, xiG, qfG = fit_gpd(ex - u)
        qG = u + qfG(qq)
        raw = _log_scale_fits(np.log(ex / u), u, qq, lambda r: u * r)
        sh = _log_scale_fits(np.log((ex - m) / (u - m)), u, qq, lambda r: m + (u - m) * r)
        out.append((name, seed, k, aG, qG) + raw + sh)
    return out
