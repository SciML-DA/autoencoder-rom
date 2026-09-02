from __future__ import annotations
import contextlib, io, os, sys, time
from typing import Optional
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datasets import (
    LEAK_THRESHOLD,
    SHIFT_THRESHOLD,
    SPECS,
    load_snapshots,
    prepare_split,
)
from models.data_driven.autoencoders import POD, AE, CAE, AEJax, CAEJax


# ── dataset specs ──────────────────────────────────────────────────────────────
# fixed hidden widths make the dense AE's parameter count essentially independent
# of the latent size, so its curve is flat by construction. scaling the widths
# keeps params proportional to n_latent, the same scaling POD has
SCALED_HIDDEN = lambda n_latent: (8 * n_latent, 2 * n_latent)

# resolution and file layout live in datasets.SPECS. only the model
# hyperparameters this study chose belong here
CONV_DEFAULT = {"channels": (16, 32, 64), "kernel_size": 3, "stride": 2, "pad": 1}

DATASETS = {
    "bl": {
        "spec": SPECS["bl"],
        "hidden": SCALED_HIDDEN,
        "conv": CONV_DEFAULT,
        "batch_size": 32,
    },
    "circle": {
        "spec": SPECS["circle"],
        "hidden": SCALED_HIDDEN,
        "conv": CONV_DEFAULT,
        "batch_size": 32,
    },
}

TRAIN_DEFAULTS = {
    "learning_rate": 1e-3,
    "n_epochs": 600,
    "weight_decay": 1e-6,
    # patience has to cover every LR decay down to min_lr, see check_schedule
    "patience": 80,
    "lr_factor": 0.5,
    "lr_patience": 10,
    "min_lr": 1e-5,
}

BACKEND = {
    "POD": "numpy",
    "AE": "torch",
    "CAE": "torch",
    "AEJax": "jax",
    "CAEJax": "jax",
}
STYLE = {
    "POD": {"color": "k", "ls": "-", "marker": "s"},
    "AE": {"color": "C0", "ls": "-", "marker": "o"},
    "AEJax": {"color": "C0", "ls": "--", "marker": "^"},
    "CAE": {"color": "C1", "ls": "-", "marker": "o"},
    "CAEJax": {"color": "C1", "ls": "--", "marker": "^"},
}


# early stopping and ReduceLROnPlateau watch the same plateau on independent
# counters, so they can silently fight. a decay needs lr_patience+1 bad epochs and
# reaching min_lr needs ceil(log(min_lr/lr)/log(lr_factor)) of them; if patience is
# smaller, the loop breaks with decays left and min_lr is dead config. the old
# 1e-3 -> 1e-6 needed 160 plateau epochs against a patience of 50
def check_schedule(train: dict) -> int:
    n_decays = int(
        np.ceil(
            np.log(train["min_lr"] / train["learning_rate"])
            / np.log(train["lr_factor"])
        )
    )
    needed = n_decays * (train["lr_patience"] + 1)
    if train["patience"] < needed:
        raise ValueError(
            f"patience={train['patience']} cannot cover the LR schedule: {n_decays} "
            f"decays x (lr_patience+1)={train['lr_patience'] + 1} = {needed} plateau "
            f"epochs to reach min_lr={train['min_lr']:.0e}. Raise patience to "
            f">= {needed}, or raise min_lr / lower lr_patience."
        )
    return needed


# ── model init ─────────────────────────────────────────────────────────────────
def pick_device(override: Optional[str] = None) -> str:
    import torch

    if override:
        return override
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_kwargs(cls, **kwargs) -> dict:
    unknown = [k for k in kwargs if not hasattr(cls, k)]
    if unknown:
        raise TypeError(f"{cls.__name__} would silently drop {sorted(unknown)}")
    return kwargs


def resolve_hidden(spec: dict, n_latent: int) -> tuple:
    h = spec["hidden"]
    return tuple(h(n_latent))


def build_models(
    n_latent: int,
    seed: int,
    spec: dict,
    device: str,
    val_fraction: float,
    n_epochs: Optional[int] = None,
) -> dict:
    train = {
        **TRAIN_DEFAULTS,
        "batch_size": spec["batch_size"],
        "seed": seed,
        "val_fraction": val_fraction,
    }
    if n_epochs is not None:
        train["n_epochs"] = n_epochs
    hidden = resolve_hidden(spec, n_latent)
    # torch's ReduceLROnPlateau uses threshold=1e-4 by default; the JAX config
    # exposes it, so set it explicitly rather than relying on two defaults agreeing
    jax_train = {**train, "threshold": 1e-4}

    return {
        "POD": POD(n_modes=n_latent, method="randomized", random_state=seed),
        "AE": AE(
            n_latent=n_latent,
            **_torch_kwargs(
                AE,
                layer_dims=hidden,
                activation_function="tanh",
                device=device,
                **train,
            ),
        ),
        "AEJax": AEJax(
            n_latent=n_latent, hidden=hidden, activation="tanh", **jax_train
        ),
        "CAE": CAE(
            n_latent=n_latent,
            **_torch_kwargs(
                CAE,
                activation_function="tanh",
                device=device,
                **spec["conv"],
                **train,
            ),
        ),
        "CAEJax": CAEJax(
            n_latent=n_latent, activation="tanh", **spec["conv"], **jax_train
        ),
    }


# ── helpers ────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def quiet(enabled: bool = True):
    """Swallow the per-epoch prints inside models.data_driven.autoencoders.ae_jax.fit_params."""
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def reconstruction_metrics(model, X: np.ndarray) -> tuple[float, float]:
    Q = model.preprocess_snapshot(X)  # zero-mean flat data
    X_hat = model.reconstruct(X)  # reconstructed flat data
    mse = float(np.mean((Q + model.Q_mean - X_hat) ** 2))
    rel = mse / float(np.mean(Q**2))
    return mse, rel


# every AE takes its early-stopping set as the last round(val_fraction * n_t)
# snapshots of whatever it is handed. giving them X_train alone put that carve on
# the tail of the training block -- adjacent in time and never audited. handing
# them train+val with a matched fraction lands it on the gap-separated val block
# without touching the encoders. POD is excluded: it has no early stopping, so the
# val block would just be free extra fitting data
def fit_data(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, float]:
    X_fit = np.concatenate([X_train, X_val], axis=1)
    return X_fit, X_val.shape[1] / X_fit.shape[1]


def evaluate_model(
    name: str,
    model,
    X_fit: np.ndarray,  # what the model is fitted on
    X_train: np.ndarray,  # what train error is measured on, val excluded
    X_test: np.ndarray,
    verbose: bool = False,
) -> dict:
    t0 = time.time()
    with quiet(not verbose):
        model.fit(X_fit)
    fit_time = time.time() - t0

    train_mse, train_rel_error = reconstruction_metrics(model, X_train)
    test_mse, test_rel_error = reconstruction_metrics(model, X_test)

    # n_epochs_run is the epoch the loop broke on, which is `patience` epochs
    # after the model actually peaked
    val_hist = list(getattr(model, "val_loss_history", []))
    best_epoch = int(np.argmin(val_hist)) + 1 if val_hist else np.nan

    return {
        "model": name,
        "backend": BACKEND[name],
        "train_mse": train_mse,
        "train_rel_error": train_rel_error,
        "test_mse": test_mse,
        "test_rel_error": test_rel_error,
        # >1 means the model reconstructs unseen snapshots worse than the ones
        # it was fitted on, which is the signal that it is memorising
        "gen_gap": test_rel_error / train_rel_error if train_rel_error > 0 else np.nan,
        "fit_time_s": float(fit_time),
        "n_params": float(getattr(model, "n_params", np.nan)),
        "n_epochs_run": float(getattr(model, "n_epochs_run", np.nan)),
        "best_epoch": float(best_epoch),
        "history_train": list(getattr(model, "loss_history", [])),
        "history_val": val_hist,
        "history_lr": list(getattr(model, "lr_history", [])),
    }


def save_loss_curves(rows: list[dict], outpath: Path, title: str) -> None:
    """Train/val loss per epoch for every AE at one latent size."""
    rows = [r for r in rows if len(r["history_train"])]
    if not rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for r in rows:
        st = STYLE[r["model"]]
        axes[0].plot(
            r["history_train"], color=st["color"], ls=st["ls"], label=r["model"]
        )
        if len(r["history_val"]):
            axes[1].plot(
                r["history_val"], color=st["color"], ls=st["ls"], label=r["model"]
            )
            if np.isfinite(r["best_epoch"]):
                axes[1].axvline(
                    r["best_epoch"] - 1, color=st["color"], ls=":", lw=0.8, alpha=0.6
                )

    for ax, lab in zip(axes, ("training loss", "validation loss")):
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def _draw_floors(floors: tuple, latents: list) -> None:
    for value, label, style in floors:
        if not (value and value > 0):
            continue
        plt.axhline(value, **style)
        plt.text(latents[0], value, f" {label}", fontsize=7, va="bottom")


def save_summary_plot(
    results: list[dict],
    outpath: Path,
    metric: str,
    title: str,
    ylabel: str = "",
    floors: tuple = (),
) -> None:
    plt.figure(figsize=(7, 4.5))
    models = [m for m in STYLE if any(r["model"] == m for r in results)]

    latents: list[int] = []
    for model in models:
        rr = [r for r in results if r["model"] == model]
        latents = sorted(set(int(r["n_latent"]) for r in rr))
        means, stds = [], []
        for n_latent in latents:
            vals = [r[metric] for r in rr if int(r["n_latent"]) == n_latent]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.asarray(means), np.asarray(stds)
        st = STYLE[model]
        plt.plot(
            latents,
            means,
            label=model,
            color=st["color"],
            ls=st["ls"],
            marker=st["marker"],
        )
        if np.any(stds > 0):
            plt.fill_between(
                latents, means - stds, means + stds, color=st["color"], alpha=0.2
            )

    _draw_floors(floors, latents)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("latent dimension")
    plt.ylabel(ylabel or metric.replace("_", " "))
    plt.title(title)
    plt.grid(alpha=0.3, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_train_vs_test(
    results: list[dict], outpath: Path, title: str, floors: tuple = ()
) -> None:
    """Both errors on one axis: the vertical gap between a pair is the overfit."""
    plt.figure(figsize=(7, 4.5))
    latents: list[int] = []
    for model in [m for m in STYLE if any(r["model"] == m for r in results)]:
        rr = [r for r in results if r["model"] == model]
        latents = sorted(set(int(r["n_latent"]) for r in rr))
        pick = lambda key, n: np.mean([r[key] for r in rr if int(r["n_latent"]) == n])
        st = STYLE[model]
        plt.plot(
            latents,
            [pick("test_rel_error", n) for n in latents],
            color=st["color"],
            ls=st["ls"],
            marker=st["marker"],
            label=model,
        )
        plt.plot(
            latents,
            [pick("train_rel_error", n) for n in latents],
            color=st["color"],
            ls=st["ls"],
            alpha=0.3,
        )

    _draw_floors(floors, latents)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("latent dimension")
    plt.ylabel("MSE / data variance")
    plt.title(title)
    plt.grid(alpha=0.3, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


CSV_HEADER = [
    "dataset",
    "model",
    "backend",
    "n_latent",
    "hidden",
    "seed",
    "train_mse",
    "train_rel_error",
    "test_mse",
    "test_rel_error",
    "gen_gap",
    "fit_time_s",
    "n_params",
    "n_epochs_run",
    "best_epoch",
    "n_x",
    "gap",
    "n_train",
    "n_val",
    "n_test",
    "test_nn_dist_median",
    "test_nn_dist_min",
    "test_mean_shift",
    "val_nn_dist_median",
    "val_nn_dist_min",
    "val_mean_shift",
    "span_ceiling",
]


# everything that can invalidate the sweep, printed before it starts
def report_split(meta: dict, latent_sizes: tuple) -> None:
    for tag, what in (("test", "test error"), ("val", "early stopping")):
        if meta[f"{tag}_nn_dist_median"] < LEAK_THRESHOLD:
            print(
                f"  WARNING: every {tag} snapshot sits within "
                f"{meta[f'{tag}_nn_dist_median']:.1%} of a training snapshot, so "
                f"{what} measures interpolation between seen states."
            )
        if meta[f"{tag}_mean_shift"] > SHIFT_THRESHOLD:
            print(
                f"  WARNING: train/{tag} temporal means differ by "
                f"{meta[f'{tag}_mean_shift']:.2e} of the {tag} variance. Every "
                f"model is charged for that offset."
            )

    ceiling = meta["span_ceiling"]
    print(
        f"  linear span ceiling = {ceiling:.4f}: the full {meta['n_train']}-dim span "
        f"of the training block leaves that fraction of the test energy "
        f"unrepresentable, so no model on this split can go much below it."
    )
    if ceiling > 0.1:
        print(
            f"  WARNING: ceiling is {ceiling:.1%}, so every result lands between it "
            f"and 1.0 and the sweep has almost no dynamic range. That is a "
            f"data-quantity problem, not a model problem."
        )
    dropped = [k for k in latent_sizes if k > meta["n_train"]]
    if dropped:
        print(f"  dropping latent sizes {dropped}: POD cannot exceed n_train.")


def run_convergence_study(
    dataset: str = "bl",
    latent_sizes=(2, 4, 8, 16, 32, 64, 128, 256),
    seeds=(0,),
    models: Optional[tuple] = None,
    n_epochs: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = False,
    outdir: str = "results/convergence",
):
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {sorted(DATASETS)}")
    spec = DATASETS[dataset]
    epochs = n_epochs if n_epochs is not None else TRAIN_DEFAULTS["n_epochs"]
    sched_epochs = check_schedule(TRAIN_DEFAULTS)

    out = Path(outdir) / dataset
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(device)

    X = load_snapshots(spec["spec"])
    X_train, X_val, X_test, meta = prepare_split(X)
    X_fit, val_fraction = fit_data(X_train, X_val)
    # n_x counts fluid points x fields, i.e. the true input width of the dense AE
    n_x = int((~np.isnan(X[0, 0])).sum() * X.shape[0])
    print(
        f"dataset={dataset} shape={X.shape} n_x={n_x} device={device} epochs={epochs} "
        f"(LR schedule needs {sched_epochs} plateau epochs) "
        f"val_fraction={val_fraction:.4f}\n  split={meta}"
    )
    report_split(meta, latent_sizes)
    latent_sizes = tuple(k for k in latent_sizes if k <= meta["n_train"])

    results = []
    for n_latent in latent_sizes:
        print(f"\n=== latent={n_latent} ===")
        rows_here = []
        for seed in seeds:
            built = build_models(
                n_latent, seed, spec, device, val_fraction, n_epochs=epochs
            )
            for name, model in built.items():
                if models is not None and name not in models:
                    continue
                # POD is deterministic given the data, so extra seeds add nothing
                if name == "POD" and seed != seeds[0]:
                    continue

                row = evaluate_model(
                    name,
                    model,
                    X_train if name == "POD" else X_fit,
                    X_train,
                    X_test,
                    verbose=verbose,
                )
                row.update(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "n_latent": int(n_latent),
                        # hyphenated so it survives a comma-separated file
                        "hidden": "-".join(
                            str(h) for h in resolve_hidden(spec, n_latent)
                        ),
                        "n_x": n_x,
                        **meta,
                    }
                )
                results.append(row)
                rows_here.append(row)
                print(
                    f"  {name:7s} test_rel={row['test_rel_error']:.3e} "
                    f"gap={row['gen_gap']:.2f} best={row['best_epoch']:.0f}"
                    f"/{row['n_epochs_run']:.0f} {row['fit_time_s']:.1f}s"
                )

        save_loss_curves(
            [r for r in rows_here if r["seed"] == seeds[0]],
            out / f"loss_curves_latent{n_latent}.png",
            title=f"{dataset} loss history | latent={n_latent}, seed={seeds[0]}",
        )

    # nothing below the span ceiling is achievable and nothing near the mean shift
    # says anything about the model, so both travel with every error plot
    floors = (
        (
            meta["span_ceiling"],
            "linear span ceiling",
            {"color": "r", "ls": "--", "lw": 1},
        ),
        (
            meta["test_mean_shift"],
            "train/test mean offset",
            {"color": "0.4", "ls": ":", "lw": 1},
        ),
    )

    save_train_vs_test(
        results,
        out / "train_vs_test_rel_error.png",
        title=f"{dataset}: train (faint) vs test (solid) relative error",
        floors=floors,
    )

    # gen_gap is deliberately not plotted: once a model fits the training block
    # to machine precision the ratio runs to 1e6 and destroys a shared log axis.
    # it stays in the CSV; read the train-vs-test overlay instead.
    for metric, ylab, fl in [
        ("test_mse", "test MSE", ()),
        ("test_rel_error", "test MSE / data variance", floors),
        ("fit_time_s", "fit time [s]", ()),
        ("n_params", "trainable parameters", ()),
    ]:
        save_summary_plot(
            results,
            out / f"{metric}_vs_latent.png",
            metric=metric,
            title=f"{dataset}: {ylab} vs latent dimension",
            ylabel=ylab,
            floors=fl,
        )

    csv_path = out / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for r in results:
            f.write(",".join(str(r.get(k, "")) for k in CSV_HEADER) + "\n")

    print(f"\nwrote {csv_path}")
    return results


if __name__ == "__main__":
    run_convergence_study(dataset="bl")
