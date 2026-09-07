"""The `bl` campaign: latent-dimension convergence on simulated snapshot data.

Unlike `april_wake` this is not a measurement campaign -- the data is the
turbulent boundary-layer DNS (`SPECS["bl"]`), read from a file rather than a
rig. It lives here for the same reason: everything that knows *which* dataset
is being studied belongs to the study, not to the library.

`experiments/bl/scripts/convergence_study.py` sweeps latent dimension across every projector.
Note it also carries a `circle` entry, so the name of this folder is narrower
than what the script can run; the circle results in `results/convergence/` are
all marked old, which is why it is filed under the dataset still in use.
"""
