#!/usr/bin/env python
"""
test_hyper_data.py
==================

Analysis of the staged hyperparameter search (``hyper_search.py``).

    uv run python experiments/april_wake/scripts/test_hyper_data.py

Writes plots and ``report.md`` to ``results/hyper_search/analysis/``. Every
section also prints its headline numbers, so the terminal output alone is a
usable summary.

The question each section answers
---------------------------------
  1. inventory      what was run, after the data is cleaned (read this first)
  2. convergence    did the random search sample enough, or is it still finding
                    better configurations
  3. vs linear      does ANY network beat the tuned closed-form PODLSE, per band
  4. capacity       score against parameter count, and the Pareto front
  5. overfitting    train/validation gap and best epoch against model size
  6. marginals      the effect of each hyperparameter, one at a time
  7. importance     which hyperparameters matter, jointly, via a surrogate model
  8. screen         the one-axis-at-a-time curves from the centre point
  9. pairs          the two-way interaction grids
 10. confirm        the multi-seed re-runs, and the only test-set numbers

Rules this script enforces, because the raw CSVs break all of them
-------------------------------------------------------------------
* Only ``results.csv`` is read. ``results.prelambda.csv`` is a backup taken
  before the ``lambda_field`` column existed and is a strict subset.
* Rows are deduplicated on the configuration key. ``hs/results.csv`` already
  contains every row from ``hs_r0``-``hs_r4``, so concatenating the six files
  double-counts roughly half of the random trials. ``min`` survives that; any
  mean, boxplot or importance fit does not.
* ``nmse_val`` is compared only *within* a band. A banded trial is scored
  against its own band-limited target, which is a different quantity. Across
  bands the analysis uses the excess over that band's tuned PODLSE, or the
  fullband score (``nmse_fullband`` on banded rows, ``nmse_test`` on fullband
  ones) where it exists.
* The reference is the *tuned* PODLSE (best of 630 per band, from
  ``podlse_sweep``), scored on the same three-way split. A network has to beat
  a tuned linear model, not a default one.
* ``nmse_test`` exists only for ``confirm`` rows, by design. Nothing earlier in
  this script selects on it.
"""

from __future__ import annotations

import ast
import os
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.join(ROOT, "results")
SOURCES = ["hs"] + [f"hs_r{i}" for i in range(5)]
PODLSE_DIRS = {"full": "fullband", "20": "b20", "30": "b30", "40": "b40", "60": "b60"}
OUT = os.path.join(RESULTS, "hyper_search", "analysis")

BANDS = ["20", "30", "40", "60", "full"]          # narrowest target first
TOP_K = 10                                        # leaderboard length
CONFIRM_PER_BAND = 4    # must match hyper_search.CONFIRM_PER_BAND
# Above this, weight decay is catastrophic for every branch (see the partial
# dependence). Section 7 is re-run below it, because a range that includes
# ruinous values makes that axis dominate importance and hides everything else.
WD_SANE = 1e-3

# Mirrors AXES_FOR in hyper_search.py, plus the axes every branch shares.
# Copied rather than imported so this script does not pull in torch.
COMMON = ["latent", "r_field", "n_delays", "weight_decay", "sensor_noise", "lr",
          "lambda_field", "band_hz"]
BRANCH_AXES = {
    "linear": COMMON,
    "mlp": COMMON + ["hidden", "activation", "dropout"],
    "cnn": COMMON + ["cnn_channels", "kernel_size", "activation", "dropout"],
    "gru": COMMON + ["gru_hidden", "gru_layers", "dropout"],
}
ALL_AXES = sorted({a for v in BRANCH_AXES.values() for a in v})
LOG_AXES = {"r_field", "n_delays", "weight_decay", "lr", "gru_hidden", "n_params"}

# The configuration identity, as hyper_search.py defines it minus `stage`.
CONFIG_KEY = ["latent", "branch", "r_field", "n_delays", "hidden", "cnn_channels",
              "kernel_size", "gru_hidden", "gru_layers", "activation", "dropout",
              "weight_decay", "sensor_noise", "lr", "band_hz", "ensemble",
              "lambda_field"]

FAMILY_COLOR = {"pod+linear": "C0", "pod+mlp": "C1", "pod+cnn": "C2", "pod+gru": "C3",
                "ae+linear": "C4", "ae+mlp": "C5", "ae+cnn": "C6", "ae+gru": "C7"}

# ══════════════════════════════════════════════════════════════════════════════

REPORT: list[str] = []


def rule(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)
    REPORT.append(f"\n## {t}\n")


def say(line=""):
    print(line)
    REPORT.append(line)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(p, ROOT)}")
    REPORT.append(f"\n![{name}]({name})\n")


# ── loading ───────────────────────────────────────────────────────────────────


def _band(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "full"
    s = str(x).strip()
    return "full" if s in ("", "None", "nan") else f"{float(s):g}"


def _tuple(x):
    try:
        v = ast.literal_eval(str(x))
    except (ValueError, SyntaxError):
        return ()
    return tuple(v) if isinstance(v, (tuple, list)) else (v,)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """All trials, cleaned and deduplicated, plus the raw concatenation."""
    frames = []
    for s in SOURCES:
        p = os.path.join(RESULTS, "hyper_search", s, "results.csv")
        if os.path.exists(p):
            f = pd.read_csv(p, dtype=str)
            f["source"] = s
            frames.append(f)
    raw = pd.concat(frames, ignore_index=True)

    d = raw.copy()
    if "lambda_field" not in d:
        d["lambda_field"] = "0.0"
    d["lambda_field"] = d["lambda_field"].fillna("0.0")
    d["band"] = d["band_hz"].map(_band)
    for k in ["r_field", "n_delays", "kernel_size", "gru_hidden", "gru_layers",
              "dropout", "weight_decay", "sensor_noise", "lr", "lambda_field",
              "ensemble", "seed", "n_params", "nmse_train", "nmse_val", "cos_val",
              "nmse_test", "nmse_fullband", "epochs_run", "best_epoch",
              "stopped_early", "seconds"]:
        if k in d:
            d[k] = pd.to_numeric(d[k], errors="coerce")
    d["family"] = d["latent"] + "+" + d["branch"]

    # A trial repeated across shard files, or within one, is the same
    # measurement. Keep one. `seed` is in the key: two seeds of one config in
    # the confirm stage are two measurements and both are kept.
    key = ["stage", *CONFIG_KEY, "seed"]
    dk = d[key].map(str)
    d = d.loc[~dk.duplicated()].reset_index(drop=True)

    # Tuple axes become width and depth, which a regression can use.
    for k in ("hidden", "cnn_channels"):
        t = d[k].map(_tuple)
        d[f"{k}_width"] = t.map(lambda v: max(v) if v else np.nan)
        d[f"{k}_depth"] = t.map(len)
    d["hidden_label"] = d["hidden"].astype(str)

    # One fullband score for every row that has one: banded rows carry it in
    # nmse_fullband, fullband rows in nmse_test.
    d["test_fullband"] = np.where(d["band"] == "full", d["nmse_test"],
                                  d["nmse_fullband"])
    d["gap"] = d["nmse_val"] - d["nmse_train"]
    return d, raw


def load_reference() -> dict:
    """Tuned PODLSE per band: the best of its sweep, on the same splits."""
    ref = {}
    for band, sub in PODLSE_DIRS.items():
        p = os.path.join(RESULTS, "podlse_sweep", sub, "results.csv")
        if not os.path.exists(p):
            continue
        r = pd.read_csv(p)
        best = r.loc[r["nmse_val"].idxmin()]
        ref[band] = dict(val=float(best["nmse_val"]),
                         n_delays=int(best["n_delays"]), r_field=int(best["r_field"]),
                         ridge=float(best["ridge"]), n=len(r),
                         test=np.nan, reproduced=None)
        # the one-shot test score from `podlse_sweep.py --confirm`, if it exists
        cp = os.path.join(RESULTS, "podlse_sweep", sub, "results.confirm.csv")
        if os.path.exists(cp):
            c = pd.read_csv(cp).iloc[0]
            ref[band]["test"] = float(c["nmse_test"])
            ref[band]["reproduced"] = bool(int(c["val_reproduced"]))
    return ref


# ── 1. inventory ──────────────────────────────────────────────────────────────


def section_inventory(d, raw, ref):
    rule("1. inventory")
    say(f"raw rows read           : {len(raw)} (from {len(SOURCES)} files)")
    say(f"after deduplication     : {len(d)}  "
        f"({len(raw) - len(d)} duplicate rows dropped)")
    say()
    say("trials per stage x band:")
    t = pd.crosstab(d["stage"], d["band"]).reindex(columns=BANDS, fill_value=0)
    say("```\n" + t.to_string() + "\n```")
    say()
    say("tuned PODLSE reference (best val of its sweep, same splits):")
    for b in BANDS:
        if b in ref:
            r = ref[b]
            t = (f"   test {r['test']:.4f}" if np.isfinite(r["test"]) else "")
            if r["reproduced"] is False:
                t += "  !! val did not reproduce -- test not usable"
            say(f"  {b:>5}: val {r['val']:.4f}   L={r['n_delays']}, r={r['r_field']}, "
                f"ridge={r['ridge']:g}   ({r['n']} configs){t}")
    missing = [b for b in BANDS if b in ref and np.isnan(ref[b]["test"])]
    if missing:
        say(f"  !! PODLSE has no test score for band(s) {missing} -- run")
        say("     `podlse_sweep.py --confirm` for them. Until then its sweep never touched")
        say("     the test block. Confirm-stage test numbers below can only be")
        say("     compared with PODLSE on *validation*, which flatters neither side")
        say("     but is not the test-set comparison. See the end of this report.")


# ── 2. convergence of the random search ───────────────────────────────────────


def section_convergence(d, ref):
    rule("2. did the random search sample enough?")
    r = d[d["stage"] == "random"].copy()
    fig, axes = plt.subplots(1, len(BANDS), figsize=(4 * len(BANDS), 3.6), sharey=False)
    for ax, b in zip(axes, BANDS):
        g = r[r["band"] == b].reset_index(drop=True)
        if g.empty:
            ax.set_visible(False)
            continue
        # Order is arbitrary after merging shards; average the running minimum
        # over random orderings so the curve is not an artefact of file order.
        rng = np.random.default_rng(0)
        v = g["nmse_val"].to_numpy()
        curves = np.array([np.minimum.accumulate(v[rng.permutation(len(v))])
                           for _ in range(200)])
        x = np.arange(1, len(v) + 1)
        ax.plot(x, curves.mean(0), color="C0", lw=1.6, label="best so far (mean)")
        ax.fill_between(x, np.percentile(curves, 10, 0), np.percentile(curves, 90, 0),
                        color="C0", alpha=0.2, label="10-90%")
        if b in ref:
            ax.axhline(ref[b]["val"], color="k", ls="--", lw=1.2, label="PODLSE (tuned)")
        ax.set(title=f"band {b}  (n={len(v)})", xlabel="random trials", xscale="log")
        ax.grid(alpha=0.3, which="both")
        half = curves.mean(0)[len(v) // 2 - 1]
        end = curves.mean(0)[-1]
        say(f"  band {b:>5}: {len(v):>4} trials   best after half {half:.4f}, "
            f"after all {end:.4f}  (second half bought {half - end:+.4f})")
    axes[0].set_ylabel("best nmse_val found")
    axes[0].legend(fontsize=7)
    fig.suptitle("random-search convergence: a curve still falling at the right "
                 "wants more trials")
    save(fig, "02_convergence.png")


# ── 3. does anything beat the tuned linear model ──────────────────────────────


def section_vs_linear(d, ref):
    rule("3. does any network beat the tuned PODLSE?")
    nc = d[d["stage"] != "confirm"].dropna(subset=["nmse_val"])
    fams = [f for f in FAMILY_COLOR if f in set(nc["family"])]
    M = np.full((len(fams), len(BANDS)), np.nan)
    for i, f in enumerate(fams):
        for j, b in enumerate(BANDS):
            g = nc[(nc["family"] == f) & (nc["band"] == b)]
            if len(g) and b in ref:
                M[i, j] = g["nmse_val"].min() - ref[b]["val"]

    fig, ax = plt.subplots(figsize=(1.3 * len(BANDS) + 2, 0.5 * len(fams) + 1.5))
    lim = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(fams)):
        for j in range(len(BANDS)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(BANDS)), [f"band {b}" for b in BANDS])
    ax.set_yticks(range(len(fams)), fams)
    fig.colorbar(im, ax=ax, label="best val - tuned PODLSE val   (<0 beats linear)")
    ax.set_title("best network found, minus the tuned linear model")
    save(fig, "03_vs_linear.png")

    say("best network per band (validation), against the tuned PODLSE:")
    for b in BANDS:
        g = nc[nc["band"] == b]
        if g.empty or b not in ref:
            continue
        w = g.loc[g["nmse_val"].idxmin()]
        dv = w["nmse_val"] - ref[b]["val"]
        verdict = "BEATS linear" if dv < 0 else "loses to linear"
        say(f"  {b:>5}: {w['family']:<11} val {w['nmse_val']:.4f} vs PODLSE "
            f"{ref[b]['val']:.4f}  ({dv:+.4f}, {verdict})  "
            f"{int(w['n_params']):,} params, stage {w['stage']}")
    # The leaderboard: top configurations per band, with the columns needed to
    # rebuild each one. Written in full to CSV; the head is printed.
    cols = ["band", "family", "nmse_val", "nmse_train", "n_params", "r_field",
            "n_delays", "hidden", "cnn_channels", "gru_hidden", "activation",
            "dropout", "weight_decay", "sensor_noise", "lr", "lambda_field",
            "best_epoch", "stage"]
    board = (nc.sort_values("nmse_val").groupby("band", sort=False).head(TOP_K)
             [cols].assign(excess=lambda t: t["nmse_val"]
                           - t["band"].map(lambda b: ref.get(b, {}).get("val"))))
    board = board.sort_values(["band", "nmse_val"],
                              key=lambda s: s.map(BANDS.index) if s.name == "band" else s)
    board.to_csv(os.path.join(OUT, "03_leaderboard.csv"), index=False)
    say()
    say(f"top 3 per band (full top {TOP_K} in 03_leaderboard.csv):")
    show = ["band", "family", "nmse_val", "excess", "n_params", "r_field", "n_delays",
            "lr", "lambda_field"]
    say("```\n" + board.groupby("band", sort=False).head(3)[show]
        .to_string(index=False, float_format=lambda v: f"{v:.4g}") + "\n```")
    say()
    say("Caveat: this is the minimum over hundreds of trials on one validation")
    say("block, so it is biased low by selection; PODLSE's minimum over 630")
    say("configs is biased the same way. Section 10 has the seed-averaged view.")


# ── 4. capacity ───────────────────────────────────────────────────────────────


def _pareto(x, y):
    o = np.argsort(x)
    keep, best = [], np.inf
    for i in o:
        if y[i] < best:
            keep.append(i)
            best = y[i]
    return np.array(keep, dtype=int)


def section_capacity(d, ref):
    rule("4. score against model size")
    nc = d[d["stage"] != "confirm"].dropna(subset=["nmse_val", "n_params"])
    fig, axes = plt.subplots(1, len(BANDS), figsize=(4.2 * len(BANDS), 4), sharey=False)
    for ax, b in zip(axes, BANDS):
        g = nc[nc["band"] == b]
        for f, gf in g.groupby("family"):
            ax.scatter(gf["n_params"], gf["nmse_val"], s=9, alpha=0.5,
                       color=FAMILY_COLOR.get(f, "0.5"), label=f)
        if len(g):
            k = _pareto(g["n_params"].to_numpy(), g["nmse_val"].to_numpy())
            p = g.iloc[k]
            ax.step(p["n_params"], p["nmse_val"], where="post", color="k", lw=1.3,
                    label="Pareto front")
        if b in ref:
            ax.axhline(ref[b]["val"], color="k", ls="--", lw=1.0, label="PODLSE")
        lo = np.nanpercentile(g["nmse_val"], 1) if len(g) else 0
        hi = np.nanpercentile(g["nmse_val"], 90) if len(g) else 1
        ax.set(xscale="log", title=f"band {b}", xlabel="parameters",
               ylim=(lo - 0.01, hi))
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("nmse_val")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("does more capacity help?  (y clipped at the 90th percentile)")
    save(fig, "04_capacity.png")

    f = nc[nc["band"] == "full"]
    if len(f):
        f = f.assign(size_bin=pd.qcut(np.log10(f["n_params"]), 5, duplicates="drop"))
        say("fullband: median and best nmse_val by parameter-count quintile")
        t = f.groupby("size_bin", observed=True)["nmse_val"].agg(["count", "median", "min"])
        t.index = [f"{10**iv.left:,.0f}-{10**iv.right:,.0f}" for iv in t.index]
        say("```\n" + t.round(4).to_string() + "\n```")


# ── 5. overfitting ────────────────────────────────────────────────────────────


def section_overfitting(d):
    rule("5. overfitting: the train/val gap and where training stopped")
    nc = d[d["stage"] != "confirm"].dropna(subset=["gap", "n_params"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
    for f, g in nc.groupby("family"):
        c = FAMILY_COLOR.get(f, "0.5")
        a1.scatter(g["n_params"], g["gap"], s=8, alpha=0.4, color=c, label=f)
        a2.scatter(g["n_params"], g["best_epoch"], s=8, alpha=0.4, color=c, label=f)
    a1.axhline(0, color="k", lw=0.8)
    a1.set(xscale="log", xlabel="parameters", ylabel="nmse_val - nmse_train",
           title="generalisation gap (larger = more overfit)")
    a2.set(xscale="log", yscale="log", xlabel="parameters", ylabel="best epoch",
           title="epoch of best validation loss (small = overfits early)")
    for a in (a1, a2):
        a.grid(alpha=0.3, which="both")
    a1.legend(fontsize=7, ncol=2)
    save(fig, "05_overfitting.png")

    t = nc.groupby("family").agg(trials=("gap", "size"), gap_median=("gap", "median"),
                                 best_epoch_median=("best_epoch", "median"),
                                 params_median=("n_params", "median"))
    say("```\n" + t.round(4).to_string() + "\n```")
    say("A network whose best epoch is in the single digits was at its best")
    say("almost immediately; everything after that is fitting noise that early")
    say("stopping then discards.")


# ── 6. marginals ──────────────────────────────────────────────────────────────


def _excess(d, ref):
    return d["nmse_val"] - d["band"].map(lambda b: ref.get(b, {}).get("val", np.nan))


def _levels(s: pd.Series, axis: str, n_bins: int = 6) -> pd.Series:
    """Group an axis into levels: as-is if discrete, log-binned if continuous."""
    if axis in ("latent", "activation", "band", "hidden_label", "cnn_channels"):
        return s.astype(str)
    u = s.dropna().unique()
    if len(u) <= 8:
        return s.map(lambda v: f"{v:g}")
    x = np.log10(s.clip(lower=1e-7)) if axis in LOG_AXES else s
    b = pd.Series(pd.qcut(x, n_bins, duplicates="drop"), index=s.index)
    to = (lambda v: 10 ** v) if axis in LOG_AXES else (lambda v: v)
    return b.map(lambda iv: f"{to(iv.left):.2g}-{to(iv.right):.2g}").astype(str)


def section_marginals(d, ref):
    rule("6. marginal effect of each hyperparameter (random stage only)")
    say("Uses only random-stage trials: they are sampled independently of one")
    say("another, so a level's distribution is not skewed toward the centre point")
    say("the way screen and pair rows are. Scored as EXCESS over that band's tuned")
    say("PODLSE, which puts the five bands on one axis; below zero beats linear.")
    r = d[d["stage"] == "random"].copy()
    r["excess"] = _excess(r, ref)
    r["hidden_label"] = r["hidden"].astype(str)
    for br, axes in BRANCH_AXES.items():
        g = r[r["branch"] == br]
        if len(g) < 20:
            continue
        ax_list = [("band" if a == "band_hz" else "hidden_label" if a == "hidden" else a)
                   for a in axes]
        n = len(ax_list)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows),
                                squeeze=False)
        for ax, a in zip(axs.ravel(), ax_list):
            lv = _levels(g[a], a)
            order = sorted(lv.dropna().unique(), key=_sort_key)
            data = [g.loc[lv == o, "excess"].dropna().to_numpy() for o in order]
            ax.boxplot(data, showfliers=False, widths=0.6)
            ax.set_xticks(range(1, len(order) + 1), order, rotation=35, fontsize=7,
                          ha="right")
            ax.axhline(0, color="k", ls="--", lw=0.9)
            ax.set_title(a, fontsize=9)
            ax.grid(alpha=0.3, axis="y")
        for ax in axs.ravel()[n:]:
            ax.set_visible(False)
        fig.suptitle(f"{br} branch: excess nmse_val over tuned PODLSE, by level "
                     f"({len(g)} random trials)")
        fig.tight_layout()
        save(fig, f"06_marginals_{br}.png")


def _sort_key(s):
    try:
        return (0, float(str(s).split("-")[0]))
    except ValueError:
        return (1, str(s))


# ── 7. importance ─────────────────────────────────────────────────────────────


def _features(g, branch):
    X = pd.DataFrame(index=g.index)
    for a in BRANCH_AXES[branch]:
        if a == "latent":
            X["latent_ae"] = (g["latent"] == "ae").astype(float)
        elif a == "band_hz":
            X["band_hz"] = g["band"].map(lambda b: 125.0 if b == "full" else float(b))
        elif a == "activation":
            for v in sorted(g["activation"].dropna().unique()):
                X[f"act_{v}"] = (g["activation"] == v).astype(float)
        elif a in ("hidden", "cnn_channels"):
            X[f"{a}_width"] = np.log2(g[f"{a}_width"])
            X[f"{a}_depth"] = g[f"{a}_depth"]
        elif a in LOG_AXES:
            X[a] = np.log10(g[a].clip(lower=1e-7))
        else:
            X[a] = g[a]
    return X


def section_importance(d, ref, max_wd=None, tag=""):
    what = "" if max_wd is None else f", weight_decay <= {max_wd:g} only"
    rule(f"7{tag and 'b'}. which hyperparameters matter (surrogate, random stage{what})")
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import PartialDependenceDisplay, permutation_importance
        from sklearn.model_selection import KFold
    except ImportError:
        say("  sklearn not installed; skipping")
        return
    if max_wd is None:
        say("Read this as how much the score MOVES over the sampled range, not")
        say("whether an axis helps: a range that includes ruinous values makes that")
        say("axis dominate. The partial-dependence plot says which way it moves.")
    say("A random forest predicts excess nmse_val from the hyperparameters, per")
    say("branch. Out-of-fold R^2 says how far to trust it: below ~0.3 the")
    say("importances are describing noise, not structure.")
    r = d[d["stage"] == "random"].copy()
    r["excess"] = _excess(r, ref)
    r = r.dropna(subset=["excess"])
    if max_wd is not None:
        r = r[r["weight_decay"] <= max_wd]
        say("Restricted to the non-catastrophic region, so the ranking reflects")
        say("what matters among configurations anyone would actually use.")

    brs = [b for b in BRANCH_AXES if (r["branch"] == b).sum() >= 40]
    fig, axs = plt.subplots(1, len(brs), figsize=(4.6 * len(brs), 4.6), squeeze=False)
    tops = {}
    for ax, br in zip(axs[0], brs):
        g = r[r["branch"] == br]
        X, y = _features(g, br).fillna(0.0), g["excess"].to_numpy()
        imps, r2s = [], []
        for trn, tst in KFold(5, shuffle=True, random_state=0).split(X):
            m = RandomForestRegressor(400, min_samples_leaf=3, random_state=0, n_jobs=-1)
            m.fit(X.iloc[trn], y[trn])
            r2s.append(m.score(X.iloc[tst], y[tst]))
            pi = permutation_importance(m, X.iloc[tst], y[tst], n_repeats=10,
                                        random_state=0, n_jobs=-1)
            imps.append(pi["importances_mean"])
        imp = pd.Series(np.mean(imps, 0), index=X.columns).sort_values()
        r2 = float(np.mean(r2s))
        ax.barh(imp.index, imp.values, color="C0")
        ax.set(title=f"{br}  (n={len(g)}, out-of-fold R$^2$={r2:.2f})",
               xlabel="permutation importance")
        ax.grid(alpha=0.3, axis="x")
        top = [str(c) for c in imp.index.tolist()[::-1][:3]]
        tops[br] = (top, X, y, r2)
        trust = "" if r2 >= 0.3 else "   (low R^2: treat as indicative only)"
        say(f"  {br:<7} R^2 {r2:.2f}   top: {', '.join(top)}{trust}")
    fig.suptitle("what moves the score, per branch")
    fig.tight_layout()
    save(fig, f"07_importance{tag}.png")

    # Partial dependence of the top three, from one model on all the data.
    fig, axs = plt.subplots(len(tops), 3, figsize=(12, 3.2 * len(tops)), squeeze=False)
    for row, (br, (top, X, y, r2)) in zip(axs, tops.items()):
        m = RandomForestRegressor(400, min_samples_leaf=3, random_state=0,
                                  n_jobs=-1).fit(X, y)
        for ax, feat in zip(row, top):
            PartialDependenceDisplay.from_estimator(m, X, [feat], ax=ax)
            ax.axhline(0, color="k", ls="--", lw=0.8)
            ax.set_title(f"{br}: {feat}", fontsize=9)
    fig.suptitle("partial dependence: the shape of each top effect "
                 "(log10 for log-scaled axes)")
    fig.tight_layout()
    save(fig, f"07_partial_dependence{tag}.png")


# ── 8. screen ─────────────────────────────────────────────────────────────────


def section_screen(d, ref):
    rule("8. screen: one axis at a time from the centre point")
    s = d[d["stage"] == "screen"].copy()
    if s.empty:
        say("  no screen rows")
        return
    s["excess"] = _excess(s, ref)
    axes_ = [a for a in s["axis"].dropna().unique()]
    cols = 4
    rows = int(np.ceil(len(axes_) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.3 * rows), squeeze=False)
    for ax, a in zip(axs.ravel(), axes_):
        g = s[s["axis"] == a]
        col = "band" if a == "band_hz" else "hidden_label" if a == "hidden" else a
        for f, gf in g.groupby("family"):
            lv = gf[col].astype(str).to_numpy()
            # a Python sort, not np.argsort: the keys are tuples, and numpy turns a
            # list of tuples into a 2-D array and sorts along the wrong axis
            o = sorted(range(len(lv)), key=lambda i: _sort_key(lv[i]))
            ax.plot(np.arange(len(o)), gf["excess"].to_numpy()[o], "o-", ms=3,
                    color=FAMILY_COLOR.get(f, "0.5"), label=f)
            ax.set_xticks(np.arange(len(o)), lv[o], rotation=35, fontsize=7,
                          ha="right")
        ax.axhline(0, color="k", ls="--", lw=0.9)
        ax.set_title(a, fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axs.ravel()[len(axes_):]:
        ax.set_visible(False)
    axs[0][0].set_ylabel("excess over PODLSE")
    axs[0][0].legend(fontsize=6)
    fig.suptitle("screen: each axis varied alone (one seed, so differences below "
                 "the seed spread are noise)")
    fig.tight_layout()
    save(fig, "08_screen.png")


# ── 9. pairs ──────────────────────────────────────────────────────────────────


def section_pairs(d, ref):
    rule("9. pairs: two-way interaction grids")
    p = d[d["stage"] == "pair"].copy()
    if p.empty:
        say("  no pair rows")
        return
    p["excess"] = _excess(p, ref)
    names = sorted(ALL_AXES, key=len, reverse=True)
    grids = [(ax, f) for (ax, f) in p.groupby(["axis", "family"]).groups]
    fig, axs = plt.subplots(1, len(grids), figsize=(4.2 * len(grids), 3.8), squeeze=False)
    for ax, (axis, fam) in zip(axs[0], grids):
        a = next(n for n in names if axis.startswith(n))
        b = axis[len(a) + 1:]
        ca = "band" if a == "band_hz" else "hidden_label" if a == "hidden" else a
        cb = "band" if b == "band_hz" else "hidden_label" if b == "hidden" else b
        g = p[(p["axis"] == axis) & (p["family"] == fam)]
        t = g.pivot_table(index=ca, columns=cb, values="excess", aggfunc="mean")
        t = t.loc[sorted(t.index, key=_sort_key), sorted(t.columns, key=_sort_key)]
        lim = np.nanmax(np.abs(t.to_numpy()))
        im = ax.imshow(t.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(t.shape[1]), [str(c) for c in t.columns], rotation=30,
                      fontsize=7)
        ax.set_yticks(range(t.shape[0]), [str(i) for i in t.index], fontsize=7)
        ax.set(xlabel=b, ylabel=a, title=f"{fam}", )
        fig.colorbar(im, ax=ax, fraction=0.046)
        say(f"  {fam:<11} {a} x {b}: best cell {np.nanmin(t.to_numpy()):+.4f}")
    fig.suptitle("pair grids: excess over PODLSE (an interaction shows as a pattern "
                 "that is not rows-plus-columns)")
    fig.tight_layout()
    save(fig, "09_pairs.png")


# ── 10. confirm ───────────────────────────────────────────────────────────────


def section_confirm(d, ref):
    rule("10. confirm: multi-seed re-runs, and the only test-set numbers")
    c = d[d["stage"] == "confirm"].copy()
    if c.empty:
        say("  no confirm rows")
        return
    seeds_key = [k for k in CONFIG_KEY if k != "ensemble"]
    ck = c[seeds_key].map(str).agg("|".join, axis=1)
    c["cfg"] = ck.map({k: i for i, k in enumerate(ck.unique())})
    agg = c.groupby("cfg").agg(
        band=("band", "first"), family=("family", "first"),
        n_delays=("n_delays", "first"), r_field=("r_field", "first"),
        lambda_field=("lambda_field", "first"), n_params=("n_params", "first"),
        seeds=("seed", "size"), val=("nmse_val", "mean"), val_sd=("nmse_val", "std"),
        test=("nmse_test", "mean"), test_sd=("nmse_test", "std"),
        test_full=("test_fullband", "mean"), test_full_sd=("test_fullband", "std"))
    agg["optimism"] = agg["test"] - agg["val"]

    yard = float(c.groupby("cfg")["nmse_test"].std().median())
    say(f"seed spread (median std of test NMSE across seeds): {yard:.4f}")
    say("  -- the yardstick. A difference smaller than ~2x this is not a result.")
    dup = agg[agg["seeds"] > agg["seeds"].min()]
    if len(dup):
        say(f"  !! {len(dup)} config(s) were confirmed more than once "
            f"(seeds {sorted(agg['seeds'].unique())}): the confirm stage picked the")
        say("     same configuration from two stages. Collapsed here; worth fixing in")
        say("     hyper_search.stage_confirm by deduplicating candidates before top-k.")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.8))
    x = 0
    ticks, labels = [], []
    for b in BANDS:
        g = agg[agg["band"] == b].sort_values("test")
        for _, r in g.iterrows():
            a1.errorbar(x, r["test"], yerr=r["test_sd"], fmt="o", ms=4,
                        color=FAMILY_COLOR.get(r["family"], "0.5"))
            ticks.append(x)
            labels.append(f"{r['family']} L{int(r['n_delays'])} r{int(r['r_field'])}")
            x += 1
        if b in ref and len(g):
            use_test = _ref_test_ok(ref, b)
            y = ref[b]["test"] if use_test else ref[b]["val"]
            a1.hlines(y, x - len(g) - 0.4, x - 0.6, color="k",
                      ls="-" if use_test else "--", lw=1.2)
            a1.text(x - len(g) / 2 - 0.5, y, f"  {b}", va="bottom", fontsize=8)
        x += 1
    a1.set_xticks(ticks, labels, rotation=70, fontsize=6)
    a1.set(ylabel="test NMSE (mean +/- sd over seeds)",
           title="confirmed configs per band  (solid: tuned PODLSE test; "
                 "dashed: its validation, where no test exists)")
    a1.grid(alpha=0.3, axis="y")

    full = agg.dropna(subset=["test_full"]).sort_values("test_full")
    a2.errorbar(np.arange(len(full)), full["test_full"], yerr=full["test_full_sd"],
                fmt="o", ms=4, color="C0")
    a2.set_xticks(np.arange(len(full)),
                  [f"{r.band} {r.family}" for r in full.itertuples()], rotation=70,
                  fontsize=6)
    a2.set(ylabel="fullband test NMSE",
           title="every confirmed config scored on the FULL field  (comparable "
                 "across bands)")
    a2.grid(alpha=0.3, axis="y")
    save(fig, "10_confirm.png")

    say()
    say("per band, best confirmed configuration (mean over seeds):")
    for b in BANDS:
        g = agg[agg["band"] == b]
        if g.empty:
            continue
        w = g.loc[g["test"].idxmin()]
        rv = ref.get(b, {}).get("val", np.nan)
        if _ref_test_ok(ref, b):
            rt = ref[b]["test"]
            vs = (f"PODLSE test {rt:.4f}, network {w['test'] - rt:+.4f} "
                  f"({'beats' if w['test'] < rt else 'loses to'} linear)")
        else:
            vs = f"PODLSE val {rv:.4f}, no PODLSE test yet"
        say(f"  {b:>5}: {w['family']:<11} test {w['test']:.4f} +/- {w['test_sd']:.4f}"
            f"   val {w['val']:.4f}   {vs}   optimism {w['optimism']:+.4f}"
            f"   fullband {w['test_full']:.4f}")
    say()
    say("`optimism` is test minus validation. It is positive because the")
    say("configurations were *chosen* on validation; its size says how much the")
    say("search overfit the validation block.")
    if len(full):
        w = full.iloc[0]
        say()
        say(f"best on the full field, any training band: {w['family']} trained at "
            f"band {w['band']}, fullband test {w['test_full']:.4f} "
            f"+/- {w['test_full_sd']:.4f}")
    return agg


def _ref_test_ok(ref, band) -> bool:
    """A PODLSE test score exists for this band and belongs to the selected model."""
    r = ref.get(band, {})
    return bool(np.isfinite(r.get("test", np.nan)) and r.get("reproduced") is not False)


def check_confirm_integrity(d, agg):
    """The confirm stage must re-run exactly the configurations it selected.

    Rebuilds the selection the way hyper_search.stage_confirm does -- best on
    validation, one row per configuration, CONFIRM_PER_BAND per band -- and
    checks every selected configuration was confirmed with every axis intact.
    Full identity rather than one suspect column: the first confirm stage
    dropped lambda_field, and the next dropped axis would not be that one.
    """
    rule("confirm integrity")
    conf = d[d["stage"] == "confirm"]
    if conf.empty:
        say("  no confirm rows")
        return
    key = [k for k in CONFIG_KEY if k != "ensemble"]

    def ids(t):
        return t[key].map(str).agg("|".join, axis=1)

    confirmed = set(ids(conf))
    nc = d[d["stage"] != "confirm"].dropna(subset=["nmse_val"])
    bad = []
    for b in BANDS:
        g = nc[nc["band"] == b].sort_values("nmse_val")
        g = g.loc[~ids(g).duplicated()].head(CONFIRM_PER_BAND)
        miss = g[~ids(g).isin(confirmed)]
        if len(miss):
            bad.append((b, len(g), miss))
    if not bad:
        say("  every selected configuration was confirmed, with every axis intact.")
        return
    say("  !! The confirm stage did NOT re-run what it selected. Selected-but-never-")
    say("     confirmed configurations, per band:")
    for b, n, miss in bad:
        say(f"       band {b:>5}: {len(miss)} of {n}")
        for _, r in miss.iterrows():
            say(f"         {r['family']:<11} L={int(r['n_delays'])} r={int(r['r_field'])} "
                f"lambda_field={r['lambda_field']:g}  (val {r['nmse_val']:.4f})")
    say("     Section 10's test numbers are therefore partly for DIFFERENT models from")
    say("     the ones chosen, and `optimism` mixes selection bias with a changed")
    say("     model. The first confirm stage dropped lambda_field; hyper_search now")
    say("     copies every axis. Re-run confirm and pull before quoting section 10.")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    d, raw = load()
    ref = load_reference()

    section_inventory(d, raw, ref)
    section_convergence(d, ref)
    section_vs_linear(d, ref)
    section_capacity(d, ref)
    section_overfitting(d)
    section_marginals(d, ref)
    section_importance(d, ref)
    section_importance(d, ref, max_wd=WD_SANE, tag="_sane")
    section_screen(d, ref)
    section_pairs(d, ref)
    agg = section_confirm(d, ref)
    check_confirm_integrity(d, agg)

    with open(os.path.join(OUT, "report.md"), "w") as fh:
        fh.write("# Hyperparameter search: analysis\n")
        fh.write("\n".join(REPORT) + "\n")
    d.to_csv(os.path.join(OUT, "trials_clean.csv"), index=False)
    rule("done")
    print(f"  {os.path.relpath(OUT, ROOT)}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
