#!/usr/bin/env python
"""
check_threads.py
================

What the job actually got, as opposed to what it asked for.

`hpc/lib.sh` exports OMP_NUM_THREADS and the BLAS-specific spellings of it, but
exporting is not the same as taking effect: numpy binds its BLAS at import and
reads the variables once, torch reads them at import too, and a wheel linked
against a threading layer nobody set a variable for will happily ignore all of
them. The only reliable check is to ask the libraries after they are loaded.

Run it inside a job, before the real work:

    uv run python scripts/check_threads.py

Exits non-zero when a library disagrees with the allocation, so a job script can
fail fast instead of spending its walltime running single-threaded.
"""

from __future__ import annotations

import os
import sys


def env_view() -> dict:
    keys = ["NCPUS", "PBS_JOBID", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "HPC_NTHREADS"]
    return {k: os.environ.get(k, "-") for k in keys}


def main() -> int:
    want = int(os.environ.get("OMP_NUM_THREADS") or 0)

    print("  environment")
    for k, v in env_view().items():
        print(f"    {k:<24} {v}")

    print("\n  as the libraries see it")
    try:
        n = len(os.sched_getaffinity(0))  # linux only
    except AttributeError:
        n = os.cpu_count()
    print(f"    schedulable cpus         {n}")

    ok = True

    import numpy as np
    print(f"    numpy                    {np.__version__}")
    try:
        # threadpoolctl ships with scikit-learn and reports the loaded BLAS
        # directly, which is the only view that cannot be talked out of.
        from threadpoolctl import threadpool_info
        for p in threadpool_info():
            got = p.get("num_threads")
            print(f"      {p.get('internal_api'):<14} {p.get('version', '?'):<10} "
                  f"threads={got}  ({os.path.basename(p.get('filepath', '?'))})")
            if want and got != want:
                ok = False
    except ImportError:
        print("      (threadpoolctl not installed -- cannot see inside the BLAS;"
              " `uv add threadpoolctl` to make this check real)")

    try:
        import torch
        print(f"    torch                    {torch.__version__}  "
              f"threads={torch.get_num_threads()}  "
              f"interop={torch.get_num_interop_threads()}  "
              f"cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"      device                 {torch.cuda.get_device_name(0)}")
        if want and torch.get_num_threads() != want:
            ok = False
    except ImportError:
        print("    torch                    not installed")

    if want and not ok:
        print(f"\n  MISMATCH: a library is not running on {want} thread(s).")
        print("  The usual cause is a variable inherited through `qsub -V` that")
        print("  hpc_threads did not clear, or a BLAS that reads a spelling")
        print("  hpc_threads does not export. Add it to hpc/lib.sh.")
        return 1

    print(f"\n  ok: everything agrees on {want or n} thread(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
