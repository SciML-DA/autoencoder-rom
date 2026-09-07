# shellcheck shell=bash
#
# hpc/lib.sh -- one place where a job script's environment is decided.
#
# Source it, call hpc_init, run your command:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
#     hpc_init
#     uv run python scripts/whatever.py ...
#
# Every job script used to carry its own copy of the thread block, the live-log
# line and the PYTHONPATH dance. Five copies drifted apart, and the drift is
# invisible until a job has been queued, has waited, has run and has produced a
# wrong number -- which is the most expensive way to find a one-line bug. This
# file is the copy.
#
# Functions, in the order hpc_init calls them:
#
#   hpc_workdir     cd to $PBS_O_WORKDIR, mkdir logs/
#   hpc_live_log    tee stdout+stderr to logs/<job>.<id>.live, tailable mid-run
#   hpc_env         PYTHONPATH, PATH, uv cache, matplotlib
#   hpc_threads     the BLAS/OpenMP thread count, resolved properly
#   hpc_report      what the job actually got, printed once
#
# hpc_gpu is separate: only the GPU jobs want it.

# ── working directory ─────────────────────────────────────────────────────────

hpc_workdir() {
    # PBS starts the job in $HOME, not where you ran qsub.
    cd "${PBS_O_WORKDIR:-$PWD}" || return 1
    mkdir -p logs
}

# ── live log ──────────────────────────────────────────────────────────────────

hpc_live_log() {
    # PBS Pro spools stdout on the compute node and copies it into logs/ only
    # when the job *ends*, so an empty logs/ mid-run is normal and an eight-hour
    # job is completely opaque until it is over. Teeing to a file under
    # $PBS_O_WORKDIR -- which is on shared storage, unlike the spool -- makes
    # progress visible:
    #
    #     tail -f logs/<jobname>.<id>.live
    #
    # PBS still writes its own .o/.e at the end; this only duplicates them.
    local id="${PBS_JOBID%%.*}"
    HPC_LIVE="logs/${PBS_JOBNAME:-job}.${id:-local}${PBS_ARRAY_INDEX:+.$PBS_ARRAY_INDEX}.live"
    exec > >(tee -a "$HPC_LIVE") 2>&1
    echo "live log: $PWD/$HPC_LIVE"
}

# ── environment ───────────────────────────────────────────────────────────────

hpc_env() {
    # No `module load tools/prod` anywhere: it puts EasyBuild's site-packages on
    # PYTHONPATH, which leaks into the uv environment and shadows its packages.
    # The venv built by setup_env_cx3.sh is self-contained. Load modules in a job
    # script only for things outside it (CUDA drivers, MPI).
    unset PYTHONPATH

    export PATH="$HOME/.local/bin:$PATH"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-${EPHEMERAL:-$HOME}/.cache/uv}"
    export MPLBACKEND=Agg

    # Without this, python buffers stdout when it is a pipe -- which it is, because
    # hpc_live_log made it one. The live log then arrives in 8 KB bursts and a
    # crashed job loses its last block entirely. This is why `tail -f` on a
    # running job used to show nothing for twenty minutes.
    export PYTHONUNBUFFERED=1

    # Deliberately no RDS_ROOT default here. Where a campaign keeps its data is
    # the campaign's business, and this file is shared by all of them -- it used
    # to hard-code the April porous-disc project path, which made every other
    # study inherit one rig's directory. Each campaign declares its own in
    # experiments/<name>/hpc/env.sh, sourced by its jobs right after hpc_init.
}

# ── threads ───────────────────────────────────────────────────────────────────

# Number of CPUs this process is actually allowed to run on.
#
# GNU nproc reports min(affinity mask, $OMP_NUM_THREADS), so it must be called
# only after the OpenMP variables are cleared -- see hpc_threads. Falls back to
# the macOS spelling so the job scripts can be exercised on a laptop.
# HPC_TEST_MASK / HPC_TEST_MACHINE are a test seam, and the only way to
# exercise the cx3 case (NCPUS=1, cpuset=8, machine=128) from a laptop. Nothing
# in a real job sets them.
_hpc_cpus_affinity() {
    local n
    n="${HPC_TEST_MASK:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 0)}"
    printf '%s' "${n:-0}"
}

# Total CPUs on the machine, for context only. Never use this as the thread
# count: on a shared node it is an order of magnitude more than the job was
# given, and oversubscribing a BLAS call is slower than running it serial.
_hpc_cpus_machine() {
    local n
    n="${HPC_TEST_MACHINE:-$(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 0)}"
    printf '%s' "${n:-0}"
}

# hpc_threads [max]
#
# Sets OMP_NUM_THREADS and every BLAS-specific spelling of it, and exports
# HPC_NTHREADS. Optional `max` caps the result -- pass it on a GPU job, where a
# dozen BLAS threads fighting the dataloader is worse than four.
#
# Which of the two available numbers to believe
# ---------------------------------------------
# There are two claims about how many CPUs this job has, and on cx3 they
# disagree: `$NCPUS` said 1 while the affinity mask said 8, for a job whose
# select line asked for 8 (job 3936942, cx3-5-17). Taking NCPUS there ran every
# SVD single-threaded on eight allocated cores.
#
# The tie-break is whether the node is cpuset-confined, which the mask itself
# tells you: a mask *narrower than the machine* means a cgroup is enforcing a
# real allocation, and it is then the authority -- it is what the kernel will
# actually schedule on, whatever PBS reports. A mask equal to the machine means
# nothing is confining the job, the mask is meaningless, and NCPUS is the only
# statement of what we are entitled to.
#
#   confined (mask < machine)   -> the mask
#   unconfined (mask == machine) -> NCPUS, else the mask
#   always                       -> clamp to the mask, never exceed it
#
# Both failure modes are then covered: a stale NCPUS cannot starve a confined
# job, and a missing cpuset cannot let an unconfined one take the whole node.
hpc_threads() {
    local cap="${1:-0}"

    # Clear the inherited values *first*. `qsub -V` copies the login shell's
    # environment into the job, and nproc honours OMP_NUM_THREADS, so a stale
    # OMP_NUM_THREADS=1 from an interactive session makes every source agree on
    # 1 and the job runs single-threaded on a node it asked 8 cores for.
    unset OMP_NUM_THREADS OMP_THREAD_LIMIT OPENBLAS_NUM_THREADS MKL_NUM_THREADS
    unset NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS
    unset GOTO_NUM_THREADS MKL_DOMAIN_NUM_THREADS

    local alloc="${NCPUS:-${SLURM_CPUS_PER_TASK:-0}}"
    local mask machine n confined=0
    mask="$(_hpc_cpus_affinity)"
    machine="$(_hpc_cpus_machine)"

    [ "${mask:-0}" -ge 1 ] 2>/dev/null && [ "${machine:-0}" -ge 1 ] 2>/dev/null \
        && [ "$mask" -lt "$machine" ] 2>/dev/null && confined=1

    if [ "$confined" -eq 1 ]; then
        n="$mask"
    else
        n="$alloc"
        [ "${n:-0}" -lt 1 ] 2>/dev/null && n="$mask"
    fi
    [ "${mask:-0}" -ge 1 ] 2>/dev/null && [ "${n:-0}" -gt "$mask" ] 2>/dev/null && n="$mask"
    [ "${n:-0}" -lt 1 ] 2>/dev/null && n=1

    # An explicit override always wins -- for a job that is memory-bound rather
    # than CPU-bound, or for reproducing a run at a fixed thread count.
    [ -n "${HPC_MAX_THREADS:-}" ] && [ "$n" -gt "$HPC_MAX_THREADS" ] && n="$HPC_MAX_THREADS"
    [ "$cap" -ge 1 ] 2>/dev/null && [ "$n" -gt "$cap" ] && n="$cap"

    export OMP_NUM_THREADS="$n"
    export OPENBLAS_NUM_THREADS="$n"   # numpy/scipy wheels on PyPI
    export MKL_NUM_THREADS="$n"        # numpy from conda-forge / intel
    export NUMEXPR_NUM_THREADS="$n"    # pandas' eval path
    export VECLIB_MAXIMUM_THREADS="$n" # Accelerate, i.e. a laptop
    export BLIS_NUM_THREADS="$n"       # some AMD builds
    export HPC_NTHREADS="$n"
    HPC_ALLOC="$alloc" HPC_MASK="$mask" HPC_MACHINE="$machine" HPC_CONFINED="$confined"
}

# ── report ────────────────────────────────────────────────────────────────────

hpc_report() {
    echo "host      : $(hostname)"
    echo "workdir   : $PWD"
    echo "threads   : ${HPC_NTHREADS:-?}  (PBS NCPUS=${HPC_ALLOC:-?}, cpuset=${HPC_MASK:-?}, machine=${HPC_MACHINE:-?}, confined=${HPC_CONFINED:-?})"
    echo "RDS_ROOT  : ${RDS_ROOT:-unset}"

    # The ways this goes wrong, each with the command that settles it.
    if [ "${HPC_CONFINED:-0}" -eq 1 ] && [ "${HPC_ALLOC:-0}" -ge 1 ] \
       && [ "${HPC_ALLOC}" -ne "${HPC_MASK}" ]; then
        echo "  note: PBS reports NCPUS=${HPC_ALLOC} but the cgroup gives"
        echo "        ${HPC_MASK} CPUs. The cpuset is what the kernel schedules on,"
        echo "        so that is what is used. If the select= line asked for"
        echo "        ${HPC_MASK}, this is only PBS mis-reporting and nothing is lost;"
        echo "        settle it with: qstat -f \$PBS_JOBID | grep -i ncpus"
    fi
    if [ "${HPC_CONFINED:-0}" -eq 0 ] && [ "${HPC_MACHINE:-0}" -gt "${HPC_NTHREADS:-1}" ]; then
        echo "  note: no cpuset on this node, so the affinity mask is the whole"
        echo "        machine. Using the PBS allocation to avoid oversubscribing it."
    fi
    if [ "${HPC_NTHREADS:-1}" -lt 4 ]; then
        echo "  WARNING: only ${HPC_NTHREADS} BLAS thread(s). Every SVD in this job"
        echo "           is about to run near-serial. Check the select= line, and:"
        echo "             qstat -f \$PBS_JOBID | grep -i ncpus"
    fi
    if [ ! -d "${RDS_ROOT:-/nonexistent}" ]; then
        echo "  WARNING: RDS_ROOT does not exist or is not readable from this node."
        echo "           The job will fail at load time. Check the project mount."
    fi
}

# ── GPU ───────────────────────────────────────────────────────────────────────

# Only the jobs that train networks call this. `|| true` throughout: a missing
# CUDA module is normal when the driver is in the container image, and a missing
# nvidia-smi must not kill the job under `set -e`.
hpc_gpu() {
    module load CUDA/12.6.0 2>/dev/null || echo "no CUDA module; assuming the driver is in the image"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version \
                   --format=csv,noheader 2>/dev/null || true
    else
        echo "nvidia-smi not on PATH -- CPU fallback if torch agrees"
    fi
}

# ── everything ────────────────────────────────────────────────────────────────

# hpc_init [max_threads]
hpc_init() {
    hpc_workdir
    hpc_live_log
    hpc_env
    hpc_threads "${1:-0}"
    hpc_report
}
