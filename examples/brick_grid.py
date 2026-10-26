"""Static medium: spectrum with a cut and differential spectrum on a grid of K, for several omega, with the control variate
taken from an independent run of the integrated spectrum. Run: python examples/brick_grid.py"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bdmps_mc as P                                                     # noqa: E402

if __name__ == "__main__":
    chi = 10.842                                  # opacity n L
    prof = P.brick(chi)
    fs = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0])  # k / sqrt(qhat L) with qhat L = mu^2 chi:  kappa^2 = f^2 chi
    for x in (0.1, 0.5, 1.0):                     # omega/omega_c with omega_c = mu^2 chi L/2:  g_ref = i/(2 chi x)
        g = 1j / (2 * chi * x)
        t0 = time.perf_counter()
        I, eI = P.spectrum_integrated(prof, g, 400000, seed=1)
        r = P.spectrum_cut_parallel(prof, g, fs**2 * chi, 2000000, seed=2, nproc=os.cpu_count() or 1, I_inf=I, eI_inf=eI)
        print("omega/omega_c = %g  (%.1f s)   integrated over all k: %.5f +- %.5f" % (x, time.perf_counter() - t0, I, eI))
        for f, a, ea, d, ed in zip(fs, r["int_cv"], r["eint_cv"], r["dif"], r["edif"]):
            print("   k/sqrt(qhat L) < %-4g  I_K = %.5f +- %.5f    dI_K/dkappa^2 = %+.6f +- %.6f" % (f, a, ea, d, ed))
