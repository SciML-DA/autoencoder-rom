# diagnosis

**From:** `qsub hpc/diagnose.pbs` → `scripts/diagnose_sensors.py --out results/diagnosis`.
CPU only; it is all linear algebra and a lag scan.

**Purpose:** find out why sparse-sensor reconstruction plateaus at NMSE ~0.83 no
matter what is thrown at it. The sweep in `../sparse_sweep/main/` gave a very
specific signature — test NMSE flat at 0.83–0.85 across latent size 4…128,
across linear/MLP/CNN/GRU branches and across POD and autoencoder latents, while
the projection floor falls from 0.62 to 0.17, and train NMSE barely beats test.
Nothing that varies changes the answer and the model cannot fit its own training
data, which is neither overfitting nor too little capacity.

Six checks, one number each: mask audit, honest per-mode observability, the NMSE
ceiling those imply, a lag scan of the PIV/force pairing, a leak test (how much
a random split flatters the same model), and the force spectrum.

## Files

- `force_spectrum.png` — check 6: is a rig resonance eating the sensor modes.
- `notch_rejected.log` — the saved log of the run (job 3909428) that notched the
  forces at 14.4/16.5/17.0 Hz on the theory that they were rig resonance.
  **The notch was rejected:** it halved the observability, because that band is
  wake meandering (St ~ 0.075), not the rig. Kept as the record —
  `hpc/lowrank.pbs` points at this file when it says not to add `--notch`.
