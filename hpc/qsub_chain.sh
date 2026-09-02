#!/usr/bin/env bash
# Submit N copies of a job, each depending on the previous one finishing.
#
#   ./hpc/qsub_chain.sh 3 hpc/sparse_sensor_sweep.pbs
#   ./hpc/qsub_chain.sh 4 hpc/sparse_sensor_sweep.pbs -v TAG=main,LATENTS="pod ae cae"
#
# Only useful for a job that is *resumable*, which sparse_sensor_sweep.py is:
# each link reads the CSV and the autoencoder cache written by the last one and
# continues from there. Chaining a non-resumable job just runs it N times.
#
# `afterany` rather than `afterok` on purpose. A job killed at walltime exits
# non-zero, and that is exactly the case you want the next link to pick up --
# `afterok` would cancel the rest of the chain at the first timeout, which is
# the opposite of what this is for.
#
# The links queue immediately and wait, so they hold their place in the queue
# rather than starting from the back each time. Cancel the lot with
# `qdel` on the printed ids.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <n_links> <script.pbs> [extra qsub args...]" >&2
    exit 2
fi

N="$1"
SCRIPT="$2"
shift 2

prev=""
for ((i = 1; i <= N; i++)); do
    if [[ -z "$prev" ]]; then
        id=$(qsub "$@" "$SCRIPT")
    else
        id=$(qsub -W "depend=afterany:$prev" "$@" "$SCRIPT")
    fi
    echo "link $i/$N: $id"
    prev="$id"
done
