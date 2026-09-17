#!/usr/bin/env bash
# Assemble a self-contained benchmark directory: bench.py, the job scripts, the
# current dataset loaders, and both generations of the autoencoders.
#
#   ./experiments/perf/stage.sh /path/to/stage
#
# The originals come out of git at e42c924^ (the parent of the commit that
# landed the optimised versions), so the "before" side is exactly what the
# repo ran, not a reconstruction. Only the three modules the models need are
# taken: the old tools/__init__ also imports the ESN, which is irrelevant here.
set -euo pipefail

STAGE="${1:?usage: $0 <stage dir>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BEFORE="e42c924^"

mkdir -p "$STAGE/impls/ae_orig" "$STAGE/src" "$STAGE/logs"
for f in autoencoders.py autoencoders_jax.py pod_spod.py; do
    git -C "$ROOT" show "$BEFORE:src/tools/$f" > "$STAGE/impls/ae_orig/$f"
done
: > "$STAGE/impls/ae_orig/__init__.py"

# The new autoencoders import `training` and `configurable` from their parent package.
mkdir -p "$STAGE/impls/ae_new"
: > "$STAGE/impls/ae_new/__init__.py"
cp "$ROOT/src/models/data_driven/training.py" "$ROOT/src/models/data_driven/configurable.py" "$STAGE/impls/ae_new/"
rsync -a --delete --exclude __pycache__ "$ROOT/src/models/data_driven/autoencoders/" "$STAGE/impls/ae_new/autoencoders/"
rsync -a --delete --exclude __pycache__ "$ROOT/src/datasets/" "$STAGE/src/datasets/"
cp "$ROOT/experiments/perf/bench.py" "$ROOT/experiments/perf/bench_t4.slr" "$ROOT/experiments/perf/bench_a40.slr" "$STAGE/"

{
    echo "before: $(git -C "$ROOT" rev-parse "$BEFORE") (e42c924^)"
    echo "after:  $(git -C "$ROOT" rev-parse HEAD) ($(git -C "$ROOT" branch --show-current))"
    dirty="$(git -C "$ROOT" status --porcelain -- src/models/data_driven/autoencoders src/datasets)"
    echo "after tree dirty: ${dirty:-no}"
} > "$STAGE/PROVENANCE"
cat "$STAGE/PROVENANCE"
