"""Parallel version of bdmps_mc.cut: independent tasks, each with its own random stream, reduced by adding the sums.

Each task is cut.sums with a child of np.random.SeedSequence(seed).spawn(ntasks): independent, reproducible streams and no
communication between tasks (suitable for process pools, job arrays or MPI). The result does not depend on the number of processes,
only on (seed, ntasks).
"""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from . import cut


def add_sums(lst):
    out = {k: lst[0][k] for k in lst[0]}
    for d in lst[1:]:
        for k in out:
            out[k] = out[k] + d[k]
    return out


def _task(args):
    prof, gref, k2s, n, ss, batch = args
    return cut.sums(prof, gref, k2s, n, ss, batch)


def spectrum_cut_parallel(prof, gref, kap2s, nsamp, seed=0, nproc=4, ntasks=None, batch=100000, I_inf=None, eI_inf=0.0, pool=None):
    """As cut.spectrum_cut, with nsamp split into ntasks (default nproc) independent streams run on nproc processes.
    The profile must be picklable (module-level functions, not lambdas) when nproc > 1."""
    ntasks = ntasks or nproc
    children = np.random.SeedSequence(seed).spawn(ntasks)
    split = [nsamp // ntasks + (1 if i < nsamp % ntasks else 0) for i in range(ntasks)]
    args = [(prof, gref, kap2s, n, ss, batch) for n, ss in zip(split, children)]
    if nproc == 1:
        res = [_task(a) for a in args]
    elif pool is not None:
        res = list(pool.map(_task, args))
    else:
        with ProcessPoolExecutor(nproc) as ex:
            res = list(ex.map(_task, args))
    return cut.statistics(add_sums(res), I_inf, eI_inf)
