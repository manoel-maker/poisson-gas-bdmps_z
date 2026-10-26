"""The single-center average over the width (paper, Sec. IV C)."""
import numpy as np
from scipy.special import exp1

gE = np.euler_gamma


def V(beta):
    """V(beta) = E[ln(1 + beta/eps)], eps ~ Exp(1),  = ln beta + gamma_E + e^beta E1(beta)   (Re beta >= 0).
    Evaluated without cancellations for small |beta| and without overflow for large |beta|."""
    b = np.atleast_1d(np.asarray(beta, dtype=complex))
    zero = b == 0
    b = np.where(zero, 1.0, b)
    lg = np.log(b) + gE
    a = np.abs(b)
    out = np.empty_like(b)
    s = a < 1.0                                   # (1 - e^b)(ln b + gamma_E) + e^b * series of Ein
    if s.any():
        x = b[s]
        ser, term = np.zeros_like(x), np.ones_like(x)
        for k in range(1, 26):
            term = term * (-x) / k
            ser -= term / k
        out[s] = -np.expm1(x) * lg[s] + np.exp(x) * ser
    m = (~s) & (a <= 40.0)
    if m.any():
        out[m] = lg[m] + np.exp(b[m]) * exp1(b[m])
    big = a > 40.0                                # continued fraction for e^b E1(b)
    if big.any():
        x = b[big]
        cf = np.zeros_like(x)
        for k in range(30, 0, -1):
            cf = k * k / (x + 2 * k + 1 - cf)
        out[big] = lg[big] + 1.0 / (x + 1 - cf)
    out[zero] = 0.0
    out = out.reshape(np.shape(beta))
    return out if out.ndim else complex(out)
