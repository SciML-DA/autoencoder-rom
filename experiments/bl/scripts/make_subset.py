"""
make_subset.py
==============

Pre-decimate Challenge1.1_train.h5 on the laptop so only a fraction of it has to
cross the network.

The full file is 35 GB of (5000, 148, 4000) float32 planes, but `SPECS["bl"]`
reads it at downsample=(32, 4) and max_snapshots=3000 -- 166 MB of actual
signal. Uploading the other 35 GB buys nothing and spends a third of a 100 GB
home quota.

Decimation here is the *same* boxcar mean `datasets.snapshots` uses, so a
subset written at (8, 1) and then loaded at (4, 4) reproduces loading the
original at (32, 4): means over 8 contiguous samples, averaged 4 at a time, are
means over 32 contiguous samples. Measured agreement is 1.6e-7 relative -- equal
in exact arithmetic, float32 accumulation order apart, the same ULP-level shift
`_decimate_yx` already warns about. Factors must divide exactly (4000 = 8*500 =
32*125, 148 = 4*37) or the truncation in `_boxcar` bites.

Output keeps the source's dataset names and native (Nt, Ny, Nx) axis order, so
`read_h5` reads it with no code change -- only the spec's `downsample` drops to
the residual factor.

    python experiments/bl/scripts/make_subset.py --xs 8 --ys 1                  # 4.4 GB, resolution headroom
    python experiments/bl/scripts/make_subset.py --xs 32 --ys 4 --max-snap 3000 # 166 MB, exactly what bl uses

Then on the cluster, with the subset written at (xs, ys):

    SPECS["bl"].replace(filename="Challenge1.1_train_sub.h5",
                        downsample=(32 // xs, 4 // ys))
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

# this file is experiments/bl/scripts/<name>.py, so three levels up is the
# repo root. `src` needs no insert: the editable install puts it on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from datasets.snapshots import _boxcar  # noqa: E402  the identical filter

FIELDS = ("Uplane", "Vplane", "Wplane")
COORDS = ("x", "y")


# h5py's __getitem__ is typed as Group | Dataset | Datatype; every name we touch
# is a Dataset, and narrowing once here keeps the loop free of casts
def dset(f: h5py.File, name: str) -> h5py.Dataset:
    d = f[name]
    assert isinstance(d, h5py.Dataset), f"{name} is not a dataset"
    return d


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="data/Challenge1.1_train.h5")
    p.add_argument("--dst", default="data/Challenge1.1_train_sub.h5")
    p.add_argument("--xs", type=int, default=8, help="x decimation (4000 axis)")
    p.add_argument("--ys", type=int, default=1, help="y decimation (148 axis)")
    p.add_argument("--max-snap", type=int, default=None, help="keep first N in time")
    p.add_argument("--chunk", type=int, default=200, help="snapshots read at a time")
    a = p.parse_args()

    with h5py.File(a.src, "r") as f, h5py.File(a.dst, "w") as g:
        nt_full, ny, nx = dset(f, FIELDS[0]).shape
        if nx % a.xs or ny % a.ys:
            p.error(f"({nx}, {ny}) not divisible by ({a.xs}, {a.ys}); would truncate")
        nt = min(a.max_snap or nt_full, nt_full)
        out_shape = (nt, ny // a.ys, nx // a.xs)
        print(f"{(nt_full, ny, nx)} -> {out_shape} per field", flush=True)

        for name in FIELDS:
            # streamed in time blocks: one full plane is 12 GB, the machine
            # writing the subset is the one that can least afford to hold it
            d = g.create_dataset(name, shape=out_shape, dtype=np.float32)
            for s in range(0, nt, a.chunk):
                e = min(s + a.chunk, nt)
                blk = np.asarray(dset(f, name)[s:e], dtype=np.float32)
                blk = _boxcar(_boxcar(blk, axis=1, factor=a.ys), axis=2, factor=a.xs)
                d[s:e] = blk
                print(f"  {name} {e}/{nt}", end="\r", flush=True)
            print(f"  {name} done      ")

        # grids are decimated by plain selection, not averaged: they are
        # coordinates, and a boxcar would place them off the cell centres
        for name in COORDS:
            if name in f:
                c = dset(f, name)
                step = a.xs if c.shape[-1] == nx else a.ys
                g.create_dataset(name, data=c[..., ::step])

    print(f"wrote {a.dst}  ({os.path.getsize(a.dst) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
