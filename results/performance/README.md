# performance

**From:** `perf/baseline.slr` → `perf/baseline.py --merge` → `perf/plots.py`
(timings) and `perf/regress.py capture` (golden). Measured on the Aero T4 nodes,
before any optimisation work.

**Purpose:** the reference every optimisation is measured against — what the
unoptimised models cost, and what they scored. This is the only results folder
whose figures are tracked by git, because a reference you cannot regenerate on
the same hardware has to be kept.

- `baseline/` — timings, compile cost, memory. See its own README.
- `regression/` — golden loss curves plus the run-to-run noise floor, so a perf
  change can be shown not to have moved the numbers. See its own README.
