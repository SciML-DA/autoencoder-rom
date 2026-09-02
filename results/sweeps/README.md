# sweeps

**From:** `perf/sweep.py` (`--shard`/`--merge`, run on the Aero GPU nodes).
That script is no longer in the working tree; recover it with
`git show a2d6c9e:perf/sweep.py`.

**Purpose:** hyperparameter sensitivity for the JAX autoencoders (`AEJax`,
`CAEJax`) on the `bl` dataset. One-at-a-time, not factorial: six axes —
`n_latent`, `learning_rate`, `weight_decay`, `batch_size`, `width`, `noise` —
each swept over 8 values with the rest held at the convergence study's defaults.
96 fits, 400 epochs each, seed 0.

The `noise` axis adds noise to the *training* data but scores against the clean
signal (`test_mse_clean`, `test_rel_clean`), so a model is not rewarded for
reproducing the noise it was trained on.

This is about which knobs matter. Cost measurements live in `../performance/`.
