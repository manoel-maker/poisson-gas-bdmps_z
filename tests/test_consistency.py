"""Consistency checks of bdmps_mc that need no external code. Run: python tests/test_consistency.py  (about one minute)."""
import os
import sys
import numpy as np
from scipy.special import sici

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bdmps_mc as P                                                     # noqa: E402

failures = []


def check(name, ok, msg):
    print("%-4s %s: %s" % ("ok" if ok else "FAIL", name, msg))
    if not ok:
        failures.append(name)


def glv(gam):
    """First order in opacity per unit chi, static medium, no kinematic cut (closed form)."""
    si, ci = sici(gam)
    return np.log(gam) + np.euler_gamma - 1 + (np.pi / 2 - ci * np.sin(gam) - (np.pi / 2 - si) * np.cos(gam)) / gam


def main():
    # 1. V(beta) against direct quadrature
    from scipy.integrate import quad
    for b in (0.3j, 0.05 + 1.2j, 2j, 60j):
        f = lambda e, im: (np.exp(-e) * np.log(1 + b / e)).imag if im else (np.exp(-e) * np.log(1 + b / e)).real
        q = quad(f, 0, np.inf, args=(0,), limit=500)[0] + 1j * quad(f, 0, np.inf, args=(1,), limit=500)[0]
        check("V(%s)" % b, abs(q - P.V(b)) < 1e-8, "|quadrature - closed form| = %.1e" % abs(q - P.V(b)))
    # 2. first order in opacity (quadrature of V) against the closed form
    for gam in (0.3, 3.0):
        fo = P.first_order(P.brick(1.0), 1j * gam).real
        check("first order gamma=%g" % gam, abs(fo - glv(gam)) < 1e-9, "%.12f vs %.12f" % (fo, glv(gam)))
    # 3. Moebius recursion against ln|det(1 + A G)|, G_ij = min(s_i, s_j)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        k = 1 + rng.poisson(6)
        s = np.sort(rng.random(k))
        y = 1j * 0.7 / rng.exponential(size=k)
        z, prev, acc = 0j, 0.0, 0j
        for j in range(k):
            z += s[j] - prev
            prev = s[j]
            acc += np.log(1 + y[j] * z)
            z /= 1 + y[j] * z
        worst = max(worst, abs(np.linalg.slogdet(np.eye(k) + y[:, None] * np.minimum.outer(s, s))[1] - acc.real))
    check("Moebius = Gelfand-Yaglom determinant", worst < 1e-10, "max difference %.1e" % worst)
    # 4. the integrated spectrum against the uncut limit of the cut estimator (two different estimators)
    chi, x = 10.842, 0.5
    prof, g = P.brick(chi), 1j / (2 * chi * x)
    a = P.spectrum_integrated(prof, g, 200000, seed=1)
    r = P.spectrum_cut(prof, g, [chi], 400000, seed=3)
    z2 = (a[0] - r["nc"][0]) / np.hypot(a[1], r["nc"][1])
    check("integrated vs cut estimator without cut", abs(z2) < 4, "%.5f +- %.5f vs %.5f +- %.5f (z = %+.1f)" % (a + r["nc"] + (z2,)))
    # 5. the differential integrates to the cut spectrum: I_K2 - I_K1 = int dkappa^2 dI/dkappa^2
    k2 = np.linspace(0.0, 20.0, 81)[1:]
    r = P.spectrum_cut(prof, g, k2, 400000, seed=4)
    trap = np.concatenate([[0.0], np.cumsum(0.5 * (r["dif"][1:] + r["dif"][:-1]) * np.diff(k2))])
    d = (r["int"] - r["int"][0]) - trap
    check("integral of the differential = cut spectrum", np.max(np.abs(d)) < 1e-3 * np.max(np.abs(r["int"])) + 1e-4,
          "max deviation %.1e (same samples; trapezoid error only)" % np.max(np.abs(d)))
    # 6. parallel result independent of the number of processes
    k2s = [2.0, 8.0]
    pl = P.power_law(5.0)
    r1 = P.spectrum_cut_parallel(pl, pl.gref(0.2), k2s, 40000, seed=7, nproc=1, ntasks=4)
    r2 = P.spectrum_cut_parallel(pl, pl.gref(0.2), k2s, 40000, seed=7, nproc=2, ntasks=4)
    check("parallel: 1 vs 2 processes", np.array_equal(r1["dif"], r2["dif"]), "identical results")
    print("\n%d failure(s)" % len(failures))
    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
