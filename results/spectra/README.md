# spectra

**From:** `qsub experiments/april_wake/hpc/spectra.pbs` → `experiments/april_wake/scripts/spectra.py`, one subfolder per run.

**Purpose:** the frequency-domain survey of the sparse-sensor problem — what the
load cells carry, what the flow carries, and at which frequencies the two are
linearly related.

It exists because `../diagnosis/` answers "how much of the field can twelve load
cells explain" with a single broadband number (NMSE ~0.83), and one number
cannot tell apart two situations that call for opposite next steps: sensors that
are uninformative at every frequency, versus sensors that are informative in a
narrow band and blind everywhere else, whose signal the broadband average
dilutes to nothing. Coherence tells them apart frequency by frequency.

It also produces the PIV/force lag estimate that phase 2 of `experiments/april_wake/hpc/run_study.sh`
has to be told (`FORCE_LAG`), from cross-correlation and cross-spectral group
delay rather than from refitting an estimator.

Run it *before* the big sweep: it costs an hour of CPU and it decides both what
lag to use and whether a broadband reconstruction is worth attempting at all.
Everything is computed on the training split.
