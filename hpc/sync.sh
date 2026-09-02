#!/usr/bin/env bash
# Sync code up / results down between laptop and the Aero Linux environment.
#   ./hpc/sync.sh push        code -> aero
#   ./hpc/sync.sh pull        results -> laptop (skips the autoencoder cache)
#   ./hpc/sync.sh pull-videos just video packs and animations
#   ./hpc/sync.sh data      one-time upload of data/ (resumable)
#   ./hpc/sync.sh quota     check home usage against the 100 GB soft quota
# Requires the `aero` Host block from hpc/ssh_config.example in ~/.ssh/config.
set -euo pipefail

# Two clusters, selected with SYNC_TARGET (default aero, so existing use is
# unchanged):
#
#   ./hpc/sync.sh push                 -> aero
#   SYNC_TARGET=cx3 ./hpc/sync.sh push -> cx3
#
# They are separate systems with separate home directories -- nothing carries
# over between them. On cx3 the experiment data already lives on RDS, so `data`
# and `verify` are not needed there; only `push` is.
case "${SYNC_TARGET:-aero}" in
  aero)
    REMOTE="aero"
    REMOTE_ROOT="/home/ljc124/autoencoder-rom"
    REMOTE_DATA="/home/ljc124/data"
    ;;
  cx3)
    REMOTE="cx3"
    REMOTE_ROOT="/rds/general/user/ljc124/home/autoencoder-rom"
    # $EPHEMERAL is 30-day scratch; staging area for anything derived
    REMOTE_DATA="/rds/general/user/ljc124/ephemeral/data"
    ;;
  *)
    echo "unknown SYNC_TARGET '${SYNC_TARGET}' (want aero or cx3)" >&2
    exit 1
    ;;
esac
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# rsync filter semantics differ from git's, so this list is deliberately separate
# from .gitignore rather than derived from it.
EXCLUDES=(
    --exclude '.git/'
    --exclude '.venv/'
    --exclude '__pycache__/'
    --exclude '*.py[cod]'
    --exclude '*.egg-info/'
    --exclude '.ipynb_checkpoints/'
    --exclude '.DS_Store'
    --exclude 'data/'
    --exclude 'results/'
    --exclude 'wandb/'
    --exclude 'logs/'
)

case "${1:-}" in
  push)
    rsync -rvlt -hP "${EXCLUDES[@]}" "${LOCAL_ROOT}/" "${REMOTE}:${REMOTE_ROOT}/"
    ;;
  pull)
    # `cache/` holds fitted-autoencoder checkpoints, which are large,
    # regenerable and useless off the cluster -- a full sweep leaves several GB
    # of them. `--dry-run` first if you want to see what is coming.
    rsync -rvlt -hP --exclude 'cache/' \
        "${REMOTE}:${REMOTE_ROOT}/results/" "${LOCAL_ROOT}/results/"
    # Job logs come down too. They are a few hundred KB of text against
    # gigabytes of results, and they are the only record of *why* a run came out
    # the way it did -- which config it was actually given, which warnings fired,
    # where it died. A results tree with no logs is a set of numbers you cannot
    # interrogate. Note `logs/` stays in EXCLUDES above, so this is one-way:
    # down from the cluster, never up, and a push can never clobber them.
    rsync -rvlt -hP "${REMOTE}:${REMOTE_ROOT}/logs/" "${LOCAL_ROOT}/logs/"
    ;;
  pull-logs)
    # Just the logs, for when a job is still running and you want to read the
    # live tee without waiting for the results. Safe mid-run: rsync copies
    # whatever has been flushed so far.
    rsync -rvlt -hP "${REMOTE}:${REMOTE_ROOT}/logs/" "${LOCAL_ROOT}/logs/"
    ;;
  pull-videos)
    # Just the video packs and rendered animations. A few tens of MB against
    # potentially gigabytes for the whole results tree, so this is the one to
    # use over a slow connection when all you want is the footage.
    rsync -rvlt -hP --prune-empty-dirs \
        --include '*/' \
        --include 'video_pack.npz' --include '*.mp4' --include '*.gif' \
        --exclude '*' \
        "${REMOTE}:${REMOTE_ROOT}/results/" "${LOCAL_ROOT}/results/"
    ;;
  data)
    # macOS ships openrsync (2.6.9-compatible), which has --append but NOT
    # --append-verify, and exits 0 after printing usage when handed an unknown
    # flag -- a silent no-op. Probe instead of assuming, and checksum after,
    # since plain --append trusts that the remote prefix matches.
    if rsync --help 2>&1 | grep -q -- '--append-verify'; then
        RESUME=(--append-verify)
    else
        RESUME=(--append --partial)
    fi
    rsync -rvlt -hP "${RESUME[@]}" "${LOCAL_ROOT}/data/" "${REMOTE}:${REMOTE_DATA}/"
    ;;
  verify)
    # confirms an --append resume did not splice mismatched halves together
    for f in "${@:2}"; do
        l=$(md5 -q "${LOCAL_ROOT}/data/$f" 2>/dev/null || md5sum "${LOCAL_ROOT}/data/$f" | cut -d' ' -f1)
        r=$(ssh "${REMOTE}" "md5sum '${REMOTE_DATA}/$f' | cut -d' ' -f1")
        [ "$l" = "$r" ] && echo "OK   $f" || echo "DIFF $f  local=$l remote=$r"
    done
    ;;
  quota)
    ssh "${REMOTE}" 'du -sh $HOME 2>/dev/null; quota -s 2>/dev/null | tail -3'
    ;;
  *)
    echo "usage: [SYNC_TARGET=aero|cx3] $0 {push|pull|pull-logs|pull-videos|data|verify <file>...|quota}" >&2
    echo "current target: ${REMOTE} (${REMOTE_ROOT})" >&2
    exit 1
    ;;
esac
