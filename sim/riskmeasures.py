"""
Tail risk measures and reinsurance layer premiums under three tail models fitted to the
k exceedances over a threshold u, and profile-likelihood intervals for them.

Tail models above u (conditional on Y > u), r = y/u >= 1:
  EE   (PH2 Exp+Erlang):  S(r) = r^{-mu} (1 + p mu log r)          (Pareto--log-gamma mixture)
  GPD  (on Y - u):        S(y) = (1 + xi (y - u)/sigma)^{-1/xi}
  Pareto (Hill model):    S(r) = r^{-alpha}

Closed forms (conditional on Y > u, retention d >= u, a = d/u):
  EE:     E[(Y-d)+] = u a^{1-mu}/(mu-1) * {1 + p mu (log a + 1/(mu-1))},          mu > 1
  GPD:    E[(Y-d)+] = (sigma + xi c)/(1 - xi) * (1 + xi c/sigma)^{-1/xi},  c = d-u, xi < 1
  Pareto: E[(Y-d)+] = u a^{1-alpha}/(alpha-1),                                    alpha > 1
Unconditional quantities multiply by the exceedance probability k/n.
  layer L xs d:      E[min((Y-d)+, L)] = E[(Y-d)+] - E[(Y-d-L)+]
  VaR_q:             S(VaR) = n(1-q)/k
  ES_q (TVaR):       VaR_q + (k/n) E[(Y-VaR_q)+ | Y>u] / (1-q)

Profile-likelihood intervals (conditional on u and k/n): every functional above is strictly
decreasing in the tail rate (mu, 1/xi or alpha) for fixed values of the other parameter, so the
constraint g(theta) = v has a unique root in that rate, and the profile log-likelihood for v is
obtained by maximising over the remaining parameter only.

usage:  python3 riskmeasures.py check     # verify closed forms against numerical integration
        python3 riskmeasures.py coverage  # coverage of profile intervals for VaR and ES (simulation)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import sys
import numpy as np
from scipy import optimize, integrate, stats

CRIT95 = stats.chi2.ppf(0.95, 1) / 2


# ------------------------------------------------------------------------------------------
# survival functions and stop-loss transforms (conditional on Y > u)
# ------------------------------------------------------------------------------------------
def ee_sf(y, u, mu, p):
    r = np.maximum(np.asarray(y, float) / u, 1.0)
    return r ** (-mu) * (1 + p * mu * np.log(r))


def ee_stoploss(d, u, mu, p):
    if mu <= 1:
        return np.inf
    a = d / u
    return u * a ** (1 - mu) / (mu - 1) * (1 + p * mu * (np.log(a) + 1 / (mu - 1)))


def gpd_sf(y, u, xi, sigma):
    c = np.maximum(np.asarray(y, float) - u, 0.0)
    if abs(xi) < 1e-10:
        return np.exp(-c / sigma)
    t = 1 + xi * c / sigma
    return np.where(t > 0, np.maximum(t, 1e-300) ** (-1 / xi), 0.0)


def gpd_stoploss(d, u, xi, sigma):
    if xi >= 1:
        return np.inf
    c = d - u
    if xi < 0 and c >= -sigma / xi:
        return 0.0
    return (sigma + xi * c) / (1 - xi) * gpd_sf(d, u, xi, sigma)


def par_sf(y, u, alpha):
    return np.maximum(np.asarray(y, float) / u, 1.0) ** (-alpha)


def par_stoploss(d, u, alpha):
    if alpha <= 1:
        return np.inf
    return u * (d / u) ** (1 - alpha) / (alpha - 1)


# ------------------------------------------------------------------------------------------
# generic risk measures from (sf, stoploss); `frac` = k/n is the exceedance probability
# ------------------------------------------------------------------------------------------
def var_q(sf, u, frac, q):
    target = (1 - q) / frac
    if target >= 1:
        return np.nan  # quantile below the threshold: not covered by the tail model
    hi = 2 * u
    while sf(hi) > target:
        hi *= 2
        if hi > 1e12 * u:
            return np.inf
    return optimize.brentq(lambda y: sf(y) - target, u, hi, xtol=1e-10 * u, rtol=1e-12)


def es_q(sf, sl, u, frac, q):
    v = var_q(sf, u, frac, q)
    if not np.isfinite(v):
        return v
    return v + frac * sl(v) / (1 - q)


def layer(sl, frac, d, L):
    """Expected loss per claim in the layer L xs d (unconditional), for d >= u."""
    return frac * (sl(d) - sl(d + L))


class EE:
    name = "E+E"

    def __init__(self, u, mu, p):
        self.u, self.mu, self.p = u, mu, p

    def sf(self, y):
        return ee_sf(y, self.u, self.mu, self.p)

    def sl(self, d):
        return ee_stoploss(d, self.u, self.mu, self.p)


class GPD:
    name = "GPD"

    def __init__(self, u, xi, sigma):
        self.u, self.xi, self.sigma = u, xi, sigma

    def sf(self, y):
        return gpd_sf(y, self.u, self.xi, self.sigma)

    def sl(self, d):
        return gpd_stoploss(d, self.u, self.xi, self.sigma)


class Pareto:
    name = "Hill"

    def __init__(self, u, alpha):
        self.u, self.alpha = u, alpha

    def sf(self, y):
        return par_sf(y, self.u, self.alpha)

    def sl(self, d):
        return par_stoploss(d, self.u, self.alpha)


def measure(model, frac, spec):
    """spec: ("VaR", q) | ("ES", q) | ("layer", d, L)."""
    if spec[0] == "VaR":
        return var_q(model.sf, model.u, frac, spec[1])
    if spec[0] == "ES":
        return es_q(model.sf, model.sl, model.u, frac, spec[1])
    if spec[0] == "layer":
        return layer(model.sl, frac, spec[1], spec[2])
    raise ValueError(spec)


# ------------------------------------------------------------------------------------------
# log-likelihoods of the exceedances
# ------------------------------------------------------------------------------------------
def ll_ee(x, mu, p):
    """x = log(Y/u)."""
    return len(x) * np.log(mu) - mu * x.sum() + np.log1p(p * (mu * x - 1)).sum()


def ll_gpd(z, xi, sigma):
    """z = Y - u."""
    if sigma <= 0:
        return -np.inf
    t = 1 + xi * z / sigma
    if np.any(t <= 0):
        return -np.inf
    if abs(xi) < 1e-10:
        return -len(z) * np.log(sigma) - z.sum() / sigma
    return -len(z) * np.log(sigma) - (1 + 1 / xi) * np.log(t).sum()


def ll_par(x, alpha):
    return len(x) * np.log(alpha) - alpha * x.sum()


# ------------------------------------------------------------------------------------------
# fits
# ------------------------------------------------------------------------------------------
def fit_ee(x):
    best = None
    h = 1 / x.mean()
    for ft in (0.8, 1.2, 1.8):
        for p0 in (0.05, 0.5, 0.95):
            r = optimize.minimize(lambda t: -ll_ee(x, np.exp(t[0]), t[1]), (np.log(h * ft), p0),
                                  method="L-BFGS-B", bounds=[(-8, 6), (0.0, 1.0)])
            if best is None or r.fun < best.fun:
                best = r
    return np.exp(best.x[0]), best.x[1], -best.fun


def fit_gpd(z):
    best = None
    m = z.mean()
    for xi0 in (0.1, 0.5, 0.9):
        r = optimize.minimize(lambda t: -ll_gpd(z, t[0], np.exp(t[1])), (xi0, np.log(m * (1 - xi0) + 1e-9)),
                              method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    return best.x[0], np.exp(best.x[1]), -best.fun


# ------------------------------------------------------------------------------------------
# profile-likelihood intervals for a risk measure
# ------------------------------------------------------------------------------------------
def _root_in_rate(g, v, lo, hi):
    """Solve g(rate) = v for a function decreasing in `rate` on (lo, hi); None if no root."""
    glo, ghi = g(lo), g(hi)
    if not (np.isfinite(glo) or glo == np.inf):
        return None
    if glo < v or ghi > v:
        return None
    return optimize.brentq(lambda r: g(r) - v, lo, hi, xtol=1e-12, rtol=1e-10)


def _profile_ee(x, u, frac, spec, v):
    def best_for_p(p):
        g = lambda mu: measure(EE(u, mu, p), frac, spec)
        mu = _root_in_rate(g, v, 1.0 + 1e-6 if spec[0] != "VaR" else 1e-3, 50.0)
        return -np.inf if mu is None else ll_ee(x, mu, p)
    grid = np.linspace(0, 1, 11)
    vals = [best_for_p(p) for p in grid]
    i = int(np.argmax(vals))
    if not np.isfinite(vals[i]):
        return -np.inf
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, 10)]
    r = optimize.minimize_scalar(lambda p: -best_for_p(p), bounds=(lo, hi), method="bounded",
                                 options=dict(xatol=1e-5))
    return max(vals[i], -r.fun)


def _profile_gpd(z, u, frac, spec, v):
    xi_max = 0.999 if spec[0] != "VaR" else 3.0

    def best_for_xi(xi):
        # decreasing in the rate 1/sigma  <=>  increasing in sigma; root over log sigma
        g = lambda ls: -measure(GPD(u, xi, np.exp(ls)), frac, spec)
        try:
            ls = _root_in_rate(g, -v, -12.0, 12.0)
        except ValueError:
            return -np.inf
        return -np.inf if ls is None else ll_gpd(z, xi, np.exp(ls))
    grid = np.linspace(-0.4, xi_max, 15)
    vals = [best_for_xi(t) for t in grid]
    i = int(np.argmax(vals))
    if not np.isfinite(vals[i]):
        return -np.inf
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = optimize.minimize_scalar(lambda t: -best_for_xi(t), bounds=(lo, hi), method="bounded",
                                 options=dict(xatol=1e-5))
    return max(vals[i], -r.fun)


def _profile_par(x, u, frac, spec, v):
    g = lambda a: measure(Pareto(u, a), frac, spec)
    a = _root_in_rate(g, v, 1.0 + 1e-6 if spec[0] != "VaR" else 1e-3, 50.0)
    return -np.inf if a is None else ll_par(x, a)


def profile_ci(model_name, y_exc, u, frac, spec, vhat, lmax, kmax=60):
    """95% profile-likelihood interval for the risk measure `spec`; upper limit may be inf."""
    if model_name == "E+E":
        prof = lambda v: _profile_ee(np.log(y_exc / u), u, frac, spec, v)
    elif model_name == "GPD":
        prof = lambda v: _profile_gpd(y_exc - u, u, frac, spec, v)
    else:
        prof = lambda v: _profile_par(np.log(y_exc / u), u, frac, spec, v)
    h = lambda v: lmax - prof(v) - CRIT95  # > 0 outside the interval
    lo = vhat
    for _ in range(kmax):
        lo_new = lo * 0.9
        if h(lo_new) > 0:
            lo = optimize.brentq(h, lo_new, lo, rtol=1e-6)
            break
        lo = lo_new
    else:
        lo = 0.0
    hi = vhat
    for _ in range(kmax):
        hi_new = hi * 1.15
        if h(hi_new) > 0:
            hi = optimize.brentq(h, hi, hi_new, rtol=1e-6)
            break
        hi = hi_new
    else:
        hi = np.inf
    return lo, hi


def fit_all(y_sorted, k):
    """Fit Hill/Pareto, GPD and E+E to the k largest observations of a sorted sample."""
    n = len(y_sorted)
    u = y_sorted[n - k - 1]
    ex = y_sorted[n - k:]
    x = np.log(ex / u)
    alpha = 1 / x.mean()
    xi, sig, llg = fit_gpd(ex - u)
    mu, p, lle = fit_ee(x)
    return dict(u=u, ex=ex, frac=k / n,
                models={"Hill": (Pareto(u, alpha), ll_par(x, alpha)),
                        "GPD": (GPD(u, xi, sig), llg),
                        "E+E": (EE(u, mu, p), lle)})


# ------------------------------------------------------------------------------------------
# checks
# ------------------------------------------------------------------------------------------
def check():
    u = 2.0
    for (mu, p) in [(1.7, 0.0), (1.7, 0.4), (2.5, 0.9), (1.2, 0.2)]:
        m = EE(u, mu, p)
        for d in (2.0, 5.0, 30.0):
            num = integrate.quad(lambda y: m.sf(y), d, np.inf, limit=400)[0]
            print(f"EE  mu={mu} p={p} d={d}: closed {m.sl(d):.8f}  numeric {num:.8f}")
    for (xi, s) in [(0.5, 1.0), (0.3, 2.0), (-0.2, 1.0)]:
        m = GPD(u, xi, s)
        d = 4.0
        num = integrate.quad(lambda y: m.sf(y), d, np.inf, limit=400)[0]
        print(f"GPD xi={xi} s={s} d={d}: closed {m.sl(d):.8f}  numeric {num:.8f}")
    # ES and layer by simulation for an EE tail
    rng = np.random.default_rng(1)
    mu, p, frac, q = 1.8, 0.5, 0.2, 0.995
    nsim = 4_000_000
    x = rng.gamma(np.where(rng.random(nsim) < p, 2, 1), 1 / mu)
    y = u * np.exp(x)                    # conditional on Y > u
    m = EE(u, mu, p)
    v = var_q(m.sf, u, frac, q)
    qq = (1 - q) / frac
    print(f"VaR: closed {v:.4f}  MC quantile {np.quantile(y, 1 - qq):.4f}")
    print(f"ES : closed {es_q(m.sf, m.sl, u, frac, q):.4f}  MC {y[y > v].mean():.4f}")
    d, L = 10.0, 20.0
    print(f"layer 20 xs 10: closed {layer(m.sl, frac, d, L):.5f}  "
          f"MC {frac * np.clip(y - d, 0, L).mean():.5f}")


def true_es(name, q):
    """ES_q = VaR_q + E[(Y - VaR_q)+]/(1-q), with the stop-loss transform from the true survival."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ph2sim import true_quantile
    v = true_quantile(name, q)
    if name == "PH2-Erlang":        # exact Exp+Erlang tail above u0 = 1 with mu = 2, p = 0.5
        sl = ee_stoploss(v, 1.0, 2.0, 0.5)
    elif name == "GPD":             # xi = 0.5, sigma = 1, location 0
        sl = gpd_stoploss(v, 0.0, 0.5, 1.0)
    elif name == "Loggamma":        # log Y ~ Gamma(1.5, rate 2); integrate S(e^t) e^t dt
        sl = integrate.quad(lambda t: np.exp(stats.gamma.logsf(t, 1.5, scale=0.5) + t),
                            np.log(v), np.inf, limit=200)[0]
    else:
        raise KeyError(name)
    return v + sl / (1 - q)


def _cov_job(args):
    name, k, seed, q, tv, tes = args
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ph2sim import DISTS
    rng = np.random.default_rng(seed)
    y = np.sort(DISTS[name][0](2000, rng))
    f = fit_all(y, k)
    out = {}
    for mm, (mod, ll) in f["models"].items():
        for me, tval in (("VaR", tv), ("ES", tes)):
            spec = (me, q)
            vh = measure(mod, f["frac"], spec)
            if not np.isfinite(vh):
                out[(mm, me)] = (True, np.inf)
                continue
            lo, hi = profile_ci(mm, f["ex"], f["u"], f["frac"], spec, vh, ll)
            out[(mm, me)] = (lo <= tval <= hi, hi - lo)
    return name, k, out


def coverage(R=300):
    from multiprocessing import Pool
    q = 0.995
    names, ks = ("PH2-Erlang", "Loggamma", "GPD"), (200, 500)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ph2sim import true_quantile
    truth = {nm: (true_quantile(nm, q), true_es(nm, q)) for nm in names}
    jobs = [(nm, k, 10_000 * i + 100 * j + r, q) + truth[nm]
            for i, nm in enumerate(names) for j, k in enumerate(ks) for r in range(R)]
    res = {}
    with Pool(4) as pool:
        for nm, k, out in pool.imap_unordered(_cov_job, jobs, chunksize=2):
            for key, (c, w) in out.items():
                res.setdefault((nm, k) + key, []).append((c, w))
    print(f"Coverage of 95% profile-likelihood intervals for VaR_{q} and ES_{q}, n = 2000, R = {R}")
    print("median interval width in brackets; an infinite upper limit gives an infinite width")
    print(f"{'DGP':12s} {'k':>5s} {'measure':>8s} {'true':>9s}  " +
          "  ".join(f"{m:>16s}" for m in ("Hill", "GPD", "E+E")))
    for nm in names:
        for k in ks:
            for me, tval in (("VaR", truth[nm][0]), ("ES", truth[nm][1])):
                cells = []
                for mm in ("Hill", "GPD", "E+E"):
                    v = np.array(res[(nm, k, mm, me)], dtype=float)
                    cells.append(f"{v[:, 0].mean():6.3f} [{np.median(v[:, 1]):7.2f}]")
                print(f"{nm:12s} {k:5d} {me:>8s} {tval:9.3f}  " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    {"check": check, "coverage": coverage}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
