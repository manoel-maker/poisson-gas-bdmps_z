# poisson-gas-bdmps_z

Python package `bdmps_mc`.

Monte Carlo evaluation of the BDMPS-Z medium-induced gluon spectrum in the soft-gluon limit for a Yukawa (Gyulassy–Wang)
potential, to all orders in opacity, for static and expanding media with arbitrary profiles $n(s)$, $\mu(s)$.

The method is described in the accompanying paper: M. R. Calvo, *A Poisson-gas approach to medium-induced gluon radiation*, Zenodo, doi:[10.5281/zenodo.23655009](https://doi.org/10.5281/zenodo.23655009) (2026). In short: the Yukawa dipole cross section is
written as a Laplace superposition of Gaussians, so that the medium becomes a Poisson gas of Gaussian scattering centers. Each
configuration is solved in closed form, and the spectrum is the average over configurations:

- **integrated over all k**: $\mathcal I=\mathbb E\ln|\Phi|$, with $\Phi$ computed by a Möbius recursion
  (`bdmps_mc.integrated`);
- **with a cut $|\mathbf k|<K$ and differential in $k^2$**: the master formula of the paper, for a whole grid of $K$ from a single
  run (`bdmps_mc.cut`), optionally in parallel (`bdmps_mc.parallel`).

## Requirements and installation

Python ≥ 3.10, NumPy and SciPy (see `requirements.txt`). No installation is needed: run from this folder, or add it to `PYTHONPATH`.

## Usage

```python
import bdmps_mc as P

prof = P.brick(10.842)             # static medium of opacity chi = n L; or P.power_law(chi), or P.Profile(chi_fun, m2_fun)
g = 1j / (2 * 10.842 * 0.5)        # g_ref = i mu^2 L / (2 omega)

I, eI = P.spectrum_integrated(prof, g, nsamp=400000, seed=1)            # integrated over all k
r = P.spectrum_cut(prof, g, [2.7, 10.8, 43.4], nsamp=1000000, seed=2,   # kappa^2 = K^2/mu^2 grid
                   I_inf=I, eI_inf=eI)                                  # optional control variate
print(r["int_cv"], r["eint_cv"])   # pi/(2 alpha_s C_R) omega dI/domega |_{k<K}, with errors
print(r["dif"], r["edif"])         # d/dkappa^2 of the above
```

Conventions (dimensionless, $\sigma=s/L$): `chi(sigma)` $=n(\sigma L)L$, `m2(sigma)` $=\mu(\sigma L)^2/\mu_{\rm ref}^2$,
$g_{\rm ref}=i\mu_{\rm ref}^2L/(2\omega)$, $\kappa=K/\mu_{\rm ref}$. The outputs are
$\mathcal I_K=\frac{\pi}{2\alpha_sC_R}\,\omega\frac{dI}{d\omega}\big|_{k<K}$ and $d\mathcal I_K/d\kappa^2$.
Every result comes with its statistical error. The truncation $u_{\min}$ of the log-uniform sampling is a (small) bias, discussed in the paper.

`examples/` contains short scripts; `tests/` contains consistency checks that do not depend on any external code:

```
python tests/test_consistency.py
```

## Reproducibility

Random streams are `numpy.random.default_rng` (PCG64). In `spectrum_cut_parallel` each task gets a child of
`numpy.random.SeedSequence(seed)`, so the result depends only on `(seed, ntasks)`, not on the number of processes.

## License

MIT (see `LICENSE`). NumPy and SciPy are BSD-licensed.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`): M. R. Calvo, *A Poisson-gas approach to medium-induced gluon radiation*, Zenodo, doi:[10.5281/zenodo.23655009](https://doi.org/10.5281/zenodo.23655009) (2026).
