# spectra / 4p5d_10ms_yaw_0_0_0

`scripts/spectra.py` on the zero-yaw run of the April wake experiment — the
baseline case. Defaults from `hpc/spectra.pbs`: `--r-field 16`, `--nperseg 256`,
`--max-lag 80`, `--observed 0.8281`.

`band_ceiling.png` is the one that decides what happens next: flat means the
sensor set is the answer and the honest result is that measurement; dropping at
low frequency means a band-limited claim is worth making. `lag.png` gives the
`FORCE_LAG` for phase 2. See `../README.md`.
