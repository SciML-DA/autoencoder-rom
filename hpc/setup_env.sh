#!/usr/bin/env bash
# Run ONCE on an Aero interactive node (spitfire/hurricane/typhoon), from the
# repo root. Those nodes cap you at 8 cores / 32 GB -- fine for a `uv sync`,
# not for anything else.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

# uv caches wheels in $HOME by default; torch's CUDA wheels are ~3 GB
export UV_CACHE_DIR="${HOME}/.cache/uv"

# On linux-x86_64 the default PyPI torch wheel already bundles CUDA, so no
# extra index is needed. Verify after with the check at the bottom.
uv sync --extra dev

cat <<'EOF'

Done. Add to ~/.bashrc on the HPC:

  export PATH="$HOME/.local/bin:$PATH"
  export MPLBACKEND=Agg
  export DATA_ROOT="$HOME/data"     # no $EPHEMERAL on Aero; that is an RCS concept

Then check CUDA is visible from inside a GPU job (not on the login node):

  uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
EOF
