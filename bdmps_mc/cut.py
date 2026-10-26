"""Spectrum with a cut |k| < K and spectrum differential in k^2 (cases 2 and 3 of the paper, Sec. V D), for a whole grid of K at once.

Master formula (paper, Eq. (26)), in units of L and mu_ref:

    I_K = Re E[ int_0^1 dt sum_{j: s_j > t} alpha_j (1 - exp(-kappa^2/D_j)) / (S'_j^- S'_j^+) ],
    alpha_j = g_ref m2(s_j)/eps_j,   D_j = Theta_j + S'_j^+/(g_ref S_j),   Theta_j = sum_{i > j} m2(s_i)/eps_i,   kappa = K/mu_ref,

with I_K = pi/(2 alpha_s C_R) omega dI/domega|_{k<K}; the differential is d I_K / d kappa^2, i.e.
omega dI/(domega dkappa^2) = (2 alpha_s C_R/pi) d I_K/d kappa^2.

Estimator (Mecke's formula on the marked center j, at sigma* = t + u):
    u log-uniform on [u_min, 1], t uniform on [0, 1 - u], weight chi(sigma*) (1 - u) u ln(1/u_min);
    centers between t and sigma*: Poisson(X(sigma*) - X(t)), uniform in opacity; they build (S, S') with
        S -> S + S' Delta  (free stretch),   S' -> S' + alpha S  (center);
    centers after sigma*: Poisson(X(1) - X(sigma*)), they only enter through Theta;
    centers before t: do not contribute and are not sampled.
kappa^2 only enters the last factor, so every configuration serves all kappa^2 of the grid.
u_min = 1e-9/(1 + kappa_max^2 |g_ref| max m2) truncates the region u < u_min (a bias; see the paper, Sec. V E).

Optional control variate: the same estimator without cut, whose mean I_inf is the integrated spectrum. I_inf must be supplied
as an independent Monte Carlo estimate (bdmps_mc.integrated.spectrum_integrated) with its error, which is propagated.
"""
import numpy as np


def sums(prof, gref, kap2s, nsamp, seed, batch=100000):
    """Additive sums of one run (they can be added across independent runs; see bdmps_mc.parallel).
    seed: integer or np.random.SeedSequence. Returns dict s1, s2, sx (2 x nk: [integrated, differential]), sdd (nk x nk), sn, sn2, n."""
    k2 = np.atleast_1d(np.asarray(kap2s, float))
    nk = k2.size
    rng = np.random.default_rng(seed)
    g = complex(gref)
    sg = np.linspace(0, 1, 2001)
    m2max = float(np.max(prof.m2(sg)))
    umin = 1e-9 / (1 + k2.max() * abs(g) * m2max)
    lg = np.log(1 / umin)
    s1 = np.zeros((2, nk))
    s2 = np.zeros((2, nk))
    sx = np.zeros((2, nk))
    sdd = np.zeros((nk, nk))
    sn = sn2 = 0.0
    n = 0
    while n < nsamp:
        m = min(batch, nsamp - n)
        u = umin ** rng.random(m)                               # log-uniform
        t = rng.random(m) * (1 - u)                             # emission time
        ss = t + u                                              # marked center
        Xt, Xs = prof.X(t), prof.X(ss)
        chis = prof.chi(ss)
        n1 = rng.poisson(np.maximum(Xs - Xt, 0.0))              # centers between emission and marked center
        n2 = rng.poisson(np.maximum(prof.Xtot - Xs, 0.0))       # centers after the marked center
        Theta = np.zeros(m)
        if n2.max() > 0:
            kk = n2.max()
            pos = prof.sigma(Xs[:, None] + rng.random((m, kk)) * (prof.Xtot - Xs)[:, None])
            Theta = np.where(np.arange(kk)[None, :] < n2[:, None], prof.m2(pos) / rng.exponential(size=(m, kk)), 0.0).sum(axis=1)
        alpha = g * prof.m2(ss) / rng.exponential(size=m)
        S = np.zeros(m, complex)
        Sd = np.ones(m, complex)
        prev = t.copy()
        for k in np.unique(n1):
            if k == 0:
                continue
            idx = np.where(n1 == k)[0]
            pos = prof.sigma(Xt[idx, None] + np.sort(rng.random((idx.size, k)), axis=1) * (Xs - Xt)[idx, None])
            pos = np.clip(pos, t[idx, None], ss[idx, None])
            a = g * prof.m2(pos) / rng.exponential(size=(idx.size, k))
            s_, sd_, pv = np.zeros(idx.size, complex), np.ones(idx.size, complex), t[idx].copy()
            for j in range(k):
                s_ = s_ + sd_ * (pos[:, j] - pv)                # free stretch
                pv = pos[:, j]
                sd_ = sd_ + a[:, j] * s_                        # kick: jump of the slope
            S[idx], Sd[idx], prev[idx] = s_, sd_, pv
        Su = S + Sd * (ss - prev)                               # S at the marked center
        Sdp = Sd + alpha * Su                                   # slope after the marked center
        base = chis * (1 - u) * u * lg * alpha / (Sd * Sdp)
        vnc = base.real                                         # estimator without cut
        D = Theta + Sdp / (g * Su)
        x = -k2[:, None] / D[None, :]
        vint = (base * -np.expm1(x)).real                      # (nk, m)
        vdif = (base * np.exp(x) / D).real
        for i, v in enumerate((vint, vdif)):
            s1[i] += v.sum(axis=1)
            s2[i] += (v * v).sum(axis=1)
            sx[i] += (v * vnc).sum(axis=1)
        sdd += vdif @ vdif.T
        sn += vnc.sum()
        sn2 += (vnc * vnc).sum()
        n += m
    return dict(s1=s1, s2=s2, sx=sx, sdd=sdd, sn=sn, sn2=sn2, n=n)


def statistics(sm, I_inf=None, eI_inf=0.0):
    """Means and errors from the sums. Keys: 'int', 'dif' (means for each kappa^2), 'eint', 'edif' (errors),
    'cov_dif' (covariance of the differential between kappa^2), 'nc' (mean and error without cut), 'n'.
    If I_inf (independent estimate of the integrated spectrum, error eI_inf) is given, also 'int_cv', 'dif_cv', 'eint_cv', 'edif_cv'."""
    s1, s2, sx, sdd, sn, sn2, n = (sm[k] for k in ("s1", "s2", "sx", "sdd", "sn", "sn2", "n"))
    mu = s1 / n
    var = s2 / n - mu**2
    mn = sn / n
    vn = sn2 / n - mn**2
    cov = sx / n - mu * mn
    out = dict(int=mu[0], dif=mu[1], eint=np.sqrt(np.maximum(var[0], 0) / n), edif=np.sqrt(np.maximum(var[1], 0) / n),
               cov_dif=(sdd / n - np.outer(mu[1], mu[1])) / n, nc=(mn, np.sqrt(vn / n)), n=n)
    if I_inf is not None and vn > 0:
        c = cov / vn
        cv = mu - c * (mn - I_inf)
        ecv = np.sqrt(np.maximum(var - cov**2 / vn, 0) / n + (c * eI_inf) ** 2)
        out.update(int_cv=cv[0], dif_cv=cv[1], eint_cv=ecv[0], edif_cv=ecv[1])
    return out


def spectrum_cut(prof, gref, kap2s, nsamp=1000000, seed=0, batch=100000, I_inf=None, eI_inf=0.0):
    """Integrated with a cut and differential, for every kappa^2 in kap2s, from a single run (see `statistics` for the output)."""
    return statistics(sums(prof, gref, kap2s, nsamp, seed, batch), I_inf, eI_inf)
