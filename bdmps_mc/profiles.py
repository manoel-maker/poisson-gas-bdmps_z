"""Medium profiles in dimensionless variables.

sigma = s/L in [0, 1] is the path length in units of the medium length L (measured from the production point).
A profile is given by two vectorized functions of sigma:

    chi(sigma) = n(sigma L) L                 density of scattering centers (opacity density)
    m2(sigma)  = mu(sigma L)^2 / mu_ref^2     local screening mass squared, in units of a reference mass

The opacity coordinate X(sigma) = int_0^sigma chi makes the Poisson process homogeneous with unit rate on [0, X(1)]:
centers are drawn uniformly in X and mapped back with sigma = X^{-1}(X). Gaps without medium (chi = 0) get no centers.
"""
import numpy as np
from scipy.integrate import quad


class Profile:
    """Dimensionless medium: chi(sigma), m2(sigma) (vectorized); `breaks` are discontinuities of the profile in sigma."""

    def __init__(self, chi, m2, breaks=(), ng=40001):
        self.chi, self.m2 = chi, m2
        self.breaks = tuple(sorted(c for c in breaks if 0.0 < c < 1.0))
        sg = np.unique(np.concatenate([np.linspace(0.0, 1.0, ng), self.breaks]))
        mid = 0.5 * (sg[1:] + sg[:-1])
        dX = chi(mid) * np.diff(sg)                       # midpoint rule (never evaluates at a discontinuity)
        X = np.concatenate([[0.0], np.cumsum(dX)])
        self.Xtot = X[-1]                                 # total opacity chi
        self.I1 = np.sum(dX * mid * m2(mid))              # int sigma chi m2
        self.I2 = np.sum(dX * mid**2 * m2(mid))           # int sigma^2 chi m2
        d = np.diff(X) > 0                                # drop the interior of empty stretches (X^{-1} single valued)
        keep = np.concatenate([d, [False]]) | np.concatenate([[False], d])
        self._X, self._s = X[keep], sg[keep]

    def X(self, sigma):
        """Accumulated opacity X(sigma) = int_0^sigma chi."""
        return np.interp(sigma, self._s, self._X)

    def sigma(self, X):
        """Inverse of X(sigma)."""
        return np.interp(X, self._X, self._s)

    def gref(self, x):
        """g_ref = i mu_ref^2 L/(2 omega) at omega = x * omega_c, with chi * omega_c = int dt t n mu^2 [Eq. (3.12) of arXiv:2307.06226]."""
        return 1j * self.Xtot / (2.0 * self.I1 * x)

    def kap2_cut(self, x, Rbar):
        """kappa^2 = K^2/mu_ref^2 for the kinematic cut K = omega, with chi Rbar = (3/2) int t^2 n mu^2 (static medium: Rbar x^2/2)."""
        return (2.0 / 3.0) * Rbar * x * x * self.I1**2 / (self.Xtot * self.I2)


class _Const:
    """Picklable constant function (profiles must be picklable to be sent to worker processes)."""

    def __init__(self, c):
        self.c = c

    def __call__(self, s):
        return self.c + 0.0 * np.asarray(s, float)


class _Power:
    """Picklable c (sigma + t0)^p."""

    def __init__(self, c, t0, p):
        self.c, self.t0, self.p = c, t0, p

    def __call__(self, s):
        return self.c * (np.asarray(s, float) + self.t0) ** self.p


def brick(chi):
    """Static medium of opacity chi = n L and constant screening mass."""
    return Profile(_Const(chi), _Const(1.0))


def power_law(chi, alpha=0.5, t0=0.1):
    """Expanding medium n ~ (sigma + t0)^(-alpha), mu^2 ~ (sigma + t0)^(-2 alpha), normalized to total opacity chi."""
    J0 = quad(lambda s: (s + t0) ** (-alpha), 0, 1)[0]
    return Profile(_Power(chi / J0, t0, -alpha), _Power(1.0, t0, -2.0 * alpha))
