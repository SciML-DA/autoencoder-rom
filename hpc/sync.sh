#!/usr/bin/env bash
# Sync code up / results down between laptop and the Aero Linux environment.
#   ./hpc/sync.sh push      code -> aero
#   ./hpc/sync.sh pull      results -> laptop
#   ./hpc/sync.sh data      one-time upload of data/ (resumable)
#   ./hpc/sync.sh quota     check home usage against the 100 GB soft quota
# Requires the `aero` Host block from hpc/ssh_config.example in ~/.ssh/config.
set -euo pipefail

REMOTE="aero"
REMOTE_ROOT="/home/ljc124/real-time-da"
REMOTE_DATA="/home/ljc124/data"
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
    rsync -rvlt -hP "${REMOTE}:${REMOTE_ROOT}/results/" "${LOCAL_ROOT}/results/"
    ;;
  data)
    # --append-verify makes an interrupted 34 GB upload resumable
    rsync -rvlt -hP --append-verify "${LOCAL_ROOT}/data/" "${REMOTE}:${REMOTE_DATA}/"
    ;;
  quota)
    ssh "${REMOTE}" 'du -sh $HOME 2>/dev/null; quota -s 2>/dev/null | tail -3'
    ;;
  *)
    echo "usage: $0 {push|pull|data|quota}" >&2
    exit 1
    ;;
esac
