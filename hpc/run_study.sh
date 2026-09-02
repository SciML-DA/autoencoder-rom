#!/usr/bin/env bash
#
# Submit the convergence study as a dependency chain, in the order the science
# requires rather than the order the queue would give you.
#
#     ./hpc/run_study.sh phase1              # ~2.5 h  (verify+spectra+diagnose)
#     FORCE_LAG=-25 ./hpc/run_study.sh phase2  # ~30 h of chained links
#     ./hpc/run_study.sh phase3              # the held-out-yaw test
#
# `phase1` submits THREE jobs (verify, spectra, diagnose). It is not the same
# thing as `qsub hpc/diagnose.pbs`, which submits only the last of them.
#     ./hpc/run_study.sh all                 # 1 then 2, with 2 using FORCE_LAG
#
# Add -n to print the qsub commands without submitting anything.
#
# Why two phases rather than one chain
# ------------------------------------
# Phase 1 measures the PIV/force offset. Phase 2 has to be told it. The lag scan
# put its optimum at -25 PIV samples (-100 ms) sitting exactly on the edge of a
# 25-sample causal window, which is the signature of an offset the embedding
# cannot reach rather than one it has resolved -- and -100 ms is twenty
# convection times over a disc diameter, which is an acquisition offset, not
# physics. Until that number is settled, every sweep result is a measurement of
# a misalignment, and the previous eight-hour job is exactly that.
#
# So: read logs/spectra.*.live, take the lag it prints, pass it as FORCE_LAG.
# If phase 1 says the pairing is already right, pass FORCE_LAG=0 and the phase-2
# results are directly comparable with what is in results/sparse_sweep/main/.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY=0
[[ "${1:-}" == "-n" ]] && { DRY=1; shift; }
PHASE="${1:-}"
FORCE_LAG="${FORCE_LAG:-0}"
LINKS="${LINKS:-4}"

sub() {  # sub <depends-on-jobid|""> <extra qsub args...> <script>
    local dep="$1"; shift
    # An empty array is "unbound" under `set -u` on bash 3.2, so both
    # expansions below are guarded. This is not pedantry: it is the same class
    # of bug as the unbound $EXTRA that has already killed a job on the queue,
    # and it only shows up on the path where there is no dependency -- i.e. the
    # first link of every chain.
    local args=()
    [[ -n "$dep" ]] && args+=(-W "depend=afterany:$dep")
    if [[ $DRY -eq 1 ]]; then
        echo "  qsub ${args[*]-} $*" >&2
        echo "DRYRUN.$RANDOM"
        return
    fi
    qsub ${args[@]+"${args[@]}"} "$@"
}

banner() { printf '\n\033[1m%s\033[0m\n' "$1" >&2; }

case "$PHASE" in

phase1|diagnose)
    banner "phase 1: verify -> spectra -> diagnose"
    a=$(sub "" hpc/verify.pbs);                     echo "  verify   $a" >&2
    b=$(sub "$a" hpc/spectra.pbs);                  echo "  spectra  $b" >&2
    c=$(sub "$b" hpc/diagnose.pbs);                 echo "  diagnose $c" >&2
    cat >&2 <<'MSG'

  When they finish:
    grep -A3 'cross-correlation' logs/spectra.*.live
    grep -A6 'best lag'          logs/diagnose.*.live
    open results/spectra/*/band_ceiling.png

  The band_ceiling plot is the one that decides what happens next. Flat means
  the sensor set is the answer and the honest result is that measurement.
  Dropping at low frequency means a band-limited target is worth claiming.
MSG
    ;;

phase2|sweep)
    banner "phase 2: lowrank -> sweep (x$LINKS, resumable) at FORCE_LAG=$FORCE_LAG"
    [[ "$FORCE_LAG" == "0" ]] && cat >&2 <<'MSG'
  NOTE: FORCE_LAG=0. If phase 1 found a non-zero offset, this reproduces the
        plateau you already have. Re-run with FORCE_LAG=<n> to test the fix.
MSG
    a=$(sub "" -v "FORCE_LAG=$FORCE_LAG" hpc/lowrank.pbs)
    echo "  lowrank  $a" >&2
    # The sweep is resumable: each link reads the CSV and the autoencoder cache
    # the last one wrote and continues. afterany, not afterok -- a link killed at
    # walltime exits non-zero and that is precisely the case the next must pick
    # up.
    prev="$a"
    for ((i = 1; i <= LINKS; i++)); do
        prev=$(sub "$prev" -v "FORCE_LAG=$FORCE_LAG,TAG=${TAG:-main}" \
                   hpc/sparse_sensor_sweep.pbs)
        echo "  sweep $i/$LINKS  $prev" >&2
    done
    echo >&2
    echo "  tail -f logs/sweep.*.live" >&2
    echo "  SYNC_TARGET=cx3 ./hpc/sync.sh pull" >&2
    ;;

phase3|crossyaw)
    banner "phase 3: held-out yaw (array job, the generalisation test)"
    a=$(sub "" -v "FORCE_LAG=$FORCE_LAG" hpc/sparse_sensors_crossyaw.pbs)
    echo "  crossyaw $a" >&2
    ;;

all)
    "$0" ${DRY:+-n} phase1
    FORCE_LAG="$FORCE_LAG" "$0" ${DRY:+-n} phase2
    ;;

*)
    sed -n '2,25p' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac
