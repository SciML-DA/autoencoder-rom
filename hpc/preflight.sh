#!/usr/bin/env bash
#
# Everything that can be checked without the queue, checked before the queue.
#
#     ./hpc/preflight.sh              # the fast checks
#     ./hpc/preflight.sh --full       # ...plus the synthetic end-to-end runs
#
# The last five job failures were all of a kind that this catches: a stale
# OMP_NUM_THREADS inherited through `qsub -V`, a flag that exists in one entry
# point and not another, an unbound variable under `set -u`, a probe on a masked
# point. None needed a compute node to find and all of them cost a queue wait.
#
# Run it on the cx3 login node. Exits non-zero if anything fails, so it can gate
# a submission:
#
#     ./hpc/preflight.sh && qsub experiments/april_wake/hpc/verify.pbs

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

pass=0; fail=0
ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; pass=$((pass + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=$((fail + 1)); }
rule() { printf '\n== %s ==\n' "$1"; }

# ── 1. the job scripts parse, and agree on how they set up ────────────────────

rule "job scripts"
# Jobs live in two places: hpc/ holds the ones that are not tied to a rig, and
# each campaign keeps its own under experiments/<name>/hpc/. Both are checked
# by the same rules -- a job script does not get laxer for living next to the
# data it reads.
JOBS=(hpc/*.pbs experiments/*/hpc/*.pbs)

for f in "${JOBS[@]}"; do
    if bash -n "$f" 2>/dev/null; then ok "$(basename "$f") parses"
    else bad "$(basename "$f") is not valid bash"; bash -n "$f"; fi
done

for f in "${JOBS[@]}"; do
    grep -q 'source .*hpc/lib.sh' "$f" \
        && ok "$(basename "$f") uses hpc/lib.sh" \
        || bad "$(basename "$f") has its own setup block -- it will drift"
done

# A job that writes into logs/ needs logs/ to exist before PBS opens its -o file.
[[ -d logs ]] && ok "logs/ exists" || { mkdir -p logs; ok "logs/ created"; }

# ── 2. the thread block, exercised rather than re-implemented ─────────────────

rule "thread resolution"
_threads() {  # _threads <NCPUS> -> the OMP_NUM_THREADS that comes out
    env -u OMP_NUM_THREADS -u OPENBLAS_NUM_THREADS -u MKL_NUM_THREADS \
        NCPUS="$1" bash -c \
        "set -euo pipefail; source '$REPO/hpc/lib.sh'; hpc_threads >/dev/null; \
         echo \$OMP_NUM_THREADS"
}
n1=$(OMP_NUM_THREADS=1 _threads 8)
[[ "$n1" == "8" ]] \
    && ok "a stale OMP_NUM_THREADS=1 is overridden (-> $n1)" \
    || bad "stale OMP_NUM_THREADS=1 survived: got $n1, wanted 8"

n2=$(_threads 4)
[[ "$n2" == "4" ]] \
    && ok "NCPUS=4 is honoured (-> $n2)" \
    || bad "NCPUS=4 gave $n2"

n3=$(_threads 0)
[[ "${n3:-0}" -ge 1 ]] \
    && ok "no NCPUS falls back sanely (-> $n3)" \
    || bad "no NCPUS gave $n3"

n4=$(HPC_MAX_THREADS=2 _threads 16)
[[ "$n4" == "2" ]] \
    && ok "HPC_MAX_THREADS caps the result (-> $n4)" \
    || bad "HPC_MAX_THREADS=2 gave $n4"

# ── 3. the python side agrees ─────────────────────────────────────────────────

rule "python environment"
if uv run python -c "import numpy, scipy, matplotlib" 2>/dev/null; then
    ok "numpy / scipy / matplotlib import"
else
    bad "the uv environment is broken -- run hpc/setup_env_cx3.sh"
fi

if env -u OMP_NUM_THREADS NCPUS=4 bash -c \
     "set -euo pipefail; source '$REPO/hpc/lib.sh'; hpc_threads >/dev/null; \
      uv run python scripts/check_threads.py >/dev/null 2>&1"; then
    ok "the BLAS actually runs on the allocated threads"
else
    bad "a library disagrees with the allocation -- run scripts/check_threads.py"
fi

# ── 4. every flag a job script passes exists in the script it calls ───────────

rule "job script arguments"
uv run python - <<'PY'
import os, re, shlex, subprocess, sys
repo = os.getcwd()
bad = 0
import glob
for job in sorted(glob.glob("hpc/*.pbs") + glob.glob("experiments/*/hpc/*.pbs")):
    probe = ('uv() { printf "PYARGS:%s\\n" "${*:3}"; }\n'
             f'PBS_O_WORKDIR={shlex.quote(repo)} PBS_JOBID=preflight.0 '
             'PBS_JOBNAME=preflight NCPUS=2 PBS_ARRAY_INDEX=0\n'
             f'source {shlex.quote(os.path.join(repo, job))}\n')
    p = subprocess.run(["bash", "-c", probe], capture_output=True, text=True,
                       cwd=repo, timeout=300)
    lines = [l[7:] for l in p.stdout.splitlines() if l.startswith("PYARGS:")]
    if not lines:
        print(f"  \033[33mskip\033[0m  {os.path.basename(job)}: builds no python command")
        continue
    argv = shlex.split(lines[-1])
    script = argv[0]
    if not os.path.exists(script):
        print(f"  \033[31mFAIL\033[0m  {os.path.basename(job)}: {script} does not exist")
        bad += 1
        continue
    helptext = subprocess.run([sys.executable, script, "--help"],
                              capture_output=True, text=True, timeout=180).stdout
    missing = [a for a in argv[1:]
               if a.startswith("--") and a not in helptext]
    if missing:
        print(f"  \033[31mFAIL\033[0m  {os.path.basename(job)} -> {os.path.basename(script)}: "
              f"unknown flag(s) {' '.join(missing)}")
        bad += 1
    else:
        print(f"  \033[32mok\033[0m    {job} -> {os.path.basename(script)} "
              f"({len(argv) - 1} args, all known)")
sys.exit(1 if bad else 0)
PY
[[ $? -eq 0 ]] && pass=$((pass + 1)) || fail=$((fail + 1))

# ── 5. RDS ────────────────────────────────────────────────────────────────────

rule "data"
RDS_ROOT="${RDS_ROOT:-/rds/general/project/immanuel/live/Seagate/april_experiment}"
if [[ -d "$RDS_ROOT" ]]; then
    ok "RDS_ROOT readable: $RDS_ROOT"
    n=$(ls "$RDS_ROOT" 2>/dev/null | wc -l | tr -d ' ')
    ok "  $n entries"
else
    printf '  \033[33mskip\033[0m  RDS_ROOT not mounted here (%s)\n' "$RDS_ROOT"
    printf '        expected off-cluster; a job on cx3 would fail\n'
fi

# ── 6. the slow ones ──────────────────────────────────────────────────────────

if [[ $FULL -eq 1 ]]; then
    rule "end-to-end on synthetic data (slow)"
    if VERIFY_SLOW=1 uv run python -m pytest tests/ -q; then
        ok "the full test suite"
    else
        bad "the test suite"
    fi
fi

rule "summary"
printf '  %d passed, %d failed\n' "$pass" "$fail"
if [[ $fail -eq 0 ]]; then
    printf '\n  Clear to submit:\n'
    printf '    qsub experiments/april_wake/hpc/verify.pbs\n'
    exit 0
fi
printf '\n  Fix the above before submitting.\n'
exit 1
