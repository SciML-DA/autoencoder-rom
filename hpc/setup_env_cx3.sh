#!/usr/bin/env bash
# Run ONCE on a cx3 login node, from the repo root.
#
# cx3 is a separate system from Aero: separate home, separate scheduler (PBS,
# not Slurm), and nothing carries over. This is the cx3 twin of setup_env.sh.
#
# Login nodes are shared and the banner is serious about it -- `uv sync` is a
# download plus an unpack, which is fine. Anything that trains is a batch job.
set -euo pipefail

# DO NOT `module load tools/prod` here, and do not let a module-provided Python
# be the interpreter uv builds against.
#
# The EasyBuild Python-bundle-PyPI module ships `pbr`, which registers a
# setuptools `egg_info.writers` entry point. setuptools loads the entry points
# of *every visible distribution* when it runs egg_info, so building this
# project picks up pbr, pbr/git.py does `import pkg_resources`, and modern
# setuptools no longer ships pkg_resources -- the build fails with a traceback
# that points at this project and has nothing to do with it.
#
# Clearing PYTHONPATH and using a uv-managed interpreter decouples the whole
# thing from the module system. Modules are still needed inside jobs for CUDA
# drivers, but not for building the environment.
unset PYTHONPATH

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

# torch's CUDA wheels are ~3 GB and $HOME is quota'd; $EPHEMERAL is not.
export UV_CACHE_DIR="${EPHEMERAL:-$HOME}/.cache/uv"
mkdir -p "${UV_CACHE_DIR}"

# uv downloads and manages its own CPython rather than using a module one.
#
# 3.13 specifically, not just ">=3.12". The dependency pins in pyproject are
# version-conditional, and 3.13 is the boundary:
#
#   python < 3.13   ->  torch 2.2,      jax        (CPU only)
#   python >= 3.13  ->  torch 2.9-2.13, jax[cuda12]
#
# so building against 3.12 succeeds and quietly gives you a two-year-old torch
# and a JAX that cannot see the GPU.
uv python install 3.13
uv sync --extra dev --python 3.13

cat <<'EOF'

Done. Add to ~/.bashrc on cx3:

  export PATH="$HOME/.local/bin:$PATH"
  export UV_CACHE_DIR="$EPHEMERAL/.cache/uv"
  export MPLBACKEND=Agg
  export RDS_ROOT=/rds/general/project/immanuel/live/Seagate/april_experiment

Deliberately NOT adding `module load tools/prod` -- see the comment at the top
of this script. Load modules inside job scripts, where they are needed for CUDA.

Note RDS_ROOT, not DATA_ROOT: the experiment data is read in place from the
project space by datasets/wake_experiment, not staged into $HOME like the
Challenge1.1 h5 was on Aero.

Check it works:

  unset PYTHONPATH
  uv run python -c "import numpy, scipy; print('numpy', numpy.__version__)"

And CUDA, from inside a GPU job (NOT on the login node):

  uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
EOF
