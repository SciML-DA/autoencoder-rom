"""Re-run the bl convergence study, sharded by model so it fits a deadline.

The committed results are dated 7 Aug, before JAX had CUDA. Their accuracy
columns are fine but fit_time_s for the JAX models is ~19x inflated because
JAX was running on CPU. This re-runs everything under identical settings on
the GPU so the timing column means something.

Each shard writes its own outdir; merge afterwards.
"""
import argparse, sys, os
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

SHARDS = {0: ("AE",), 1: ("CAE", "POD"), 2: ("AEJax", "CAEJax")}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    a = p.parse_args()
    from convergence_study import run_convergence_study
    models = SHARDS[a.shard]
    print(f"shard {a.shard}: models={models}", flush=True)
    run_convergence_study(dataset="bl", models=models,
                          outdir=f"results/convergence_new/shard{a.shard}")
