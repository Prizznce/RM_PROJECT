"""
EE-ARF-10 + Drift-Adaptive Conformal Gate: end-to-end experiment
=================================================================

    python scripts/run_ee_arf_experiment.py                      # spain + india
    python scripts/run_ee_arf_experiment.py --datasets spain india elec2
    python scripts/run_ee_arf_experiment.py --retune             # re-tune & re-freeze
    python scripts/run_ee_arf_experiment.py --quick              # smoke test (truncated streams)

Protocol
--------
Stage 0  Load each dataset, log class balance and the exact target definition.
Stage 1  Validation-only tuning of the proposed method (full grid alpha x gamma x p_bg),
         static-gate tau, under a duty constraint. Selected values are frozen in
         configs/ee_arf_frozen_config.json with a canonical SHA-256 (line-ending
         independent). If the file exists it is verified and reused - the test
         split never influences selection.
Stage 2  Single prequential pass on the test split for every model (cold start,
         same harness, same timing), over --seeds.
Stage 3  Sensitivity sweeps on the test split (labelled as such; not used for
         selection): conformal alpha x p_bg, static tau, w_FN.
Stage 4  Weekly-block Wilcoxon (Holm) + Friedman tests, summary tables,
         Pareto plots, dominance verdicts.

Outputs: results/ee_arf/{tables,figures,raw}/
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from src.controllers.ee_arf_controller import (  # noqa: E402
    DEFAULT_ARF_PARAMS,
    DEFAULT_HT_PARAMS,
    AlwaysARF,
    AlwaysTree,
    EEARF10Controller,
)
from src.data.datasets import load_dataset  # noqa: E402
from src.eval.ee_harness import run_prequential, run_prequential_lockstep  # noqa: E402
from src.eval.ee_metrics import block_metric  # noqa: E402
from src.stats.block_tests import friedman, pairwise_vs_reference  # noqa: E402

CONFIG_PATH = "configs/ee_arf_frozen_config.json"
PROPOSED = "EE-ARF-10 (Conformal Gate + Subsampled BG)"
STATIC = "EE-ARF-10 (Margin-Only Static Gate)"
REPLAY = "EE-ARF-10 (Dormant Replay B=200)"
LITERAL = "EE-ARF-10 (Conformal, final-error loss)"

# Grids (spec)
ALPHAS = [0.02, 0.05, 0.10, 0.15]
GAMMAS = [0.005, 0.01, 0.05]
P_BGS = [0.0, 0.10, 0.20]
W_FNS = [1, 5, 10]
TAU_STATIC_GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
TAU_INIT = 0.30
# A-priori (not tuned) asymmetric FN weight per dataset
W_FN_DEFAULT = {"spain": 1, "india": 5, "elec2": 1}


# ---------------------------------------------------------------------- #
# Reproducibility helpers
# ---------------------------------------------------------------------- #
def canonical_hash(payload: dict) -> str:
    """SHA-256 of canonical JSON - independent of file formatting / CRLF."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lib_versions() -> dict:
    import matplotlib
    import river
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "river": river.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


try:  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass


def log(msg: str = ""):
    print(msg, flush=True)


# ---------------------------------------------------------------------- #
# Model factories
# ---------------------------------------------------------------------- #
def make_proposed(cfg: dict, seed: int, name=PROPOSED, **override):
    p = dict(
        gate="conformal",
        staleness="background",
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        tau_init=TAU_INIT,
        w_fn=cfg["w_fn"],
        loss_mode="exit_error",
        p_bg=cfg["p_bg"],
        seed=seed,
    )
    p.update(override)
    return EEARF10Controller(name=name, **p)


def main_models(cfg: dict, seed: int) -> list:
    return [
        AlwaysTree("HT", seed=seed),
        AlwaysTree("HAT", seed=seed),
        AlwaysARF(2, seed=seed),
        AlwaysARF(3, seed=seed),
        AlwaysARF(10, seed=seed),
        EEARF10Controller(
            name=STATIC, gate="static", tau_static=cfg["tau_static"], staleness="background",
            p_bg=cfg["p_bg"], w_fn=cfg["w_fn"], seed=seed,
        ),
        make_proposed(cfg, seed, name=REPLAY, staleness="replay", p_bg=0.0, replay_buffer=200),
        make_proposed(cfg, seed, name=LITERAL, loss_mode="final_error"),
        make_proposed(cfg, seed),
    ]


def strip(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


# ---------------------------------------------------------------------- #
# Stage 1: validation tuning
# ---------------------------------------------------------------------- #
def select(df: pd.DataFrame, max_duty_pct: float) -> pd.Series:
    ok = df[df["duty_rho_pct"] <= max_duty_pct]
    if ok.empty:
        ok = df.nsmallest(1, "duty_rho_pct")
    # max temporal kappa; ties -> fewer trees evaluated
    return ok.sort_values(["kappa_temporal", "avg_trees_eval"], ascending=[False, True]).iloc[0]


def tune_dataset(key, ds, max_duty_pct, seed) -> tuple[dict, pd.DataFrame]:
    val = ds.val
    y0 = ds.val_prev_label
    w_fn = W_FN_DEFAULT.get(key, 1)
    rows = []
    grid = list(itertools.product(ALPHAS, GAMMAS, P_BGS))
    log(f"  [{ds.name}] conformal grid K={len(grid)} on {len(val):,} validation samples (w_FN={w_fn})")
    for a, g, pb in grid:
        cfg = {"alpha": a, "gamma": g, "p_bg": pb, "w_fn": w_fn}
        r = strip(run_prequential(make_proposed(cfg, seed), val, y0))
        rows.append({"family": "conformal", "alpha": a, "gamma": g, "p_bg": pb, "tau_static": np.nan, **r})
    dfc = pd.DataFrame(rows)
    best = select(dfc, max_duty_pct)
    sel = {
        "alpha": float(best["alpha"]),
        "gamma": float(best["gamma"]),
        "p_bg": float(best["p_bg"]),
        "w_fn": w_fn,
        "val_kappa_temporal": float(best["kappa_temporal"]),
        "val_accuracy_pct": float(best["accuracy_pct"]),
        "val_duty_rho_pct": float(best["duty_rho_pct"]),
        "val_avg_trees_eval": float(best["avg_trees_eval"]),
    }
    log(f"    -> selected alpha={sel['alpha']} gamma={sel['gamma']} p_bg={sel['p_bg']} "
        f"(val kT={sel['val_kappa_temporal']:.4f}, rho={sel['val_duty_rho_pct']:.1f}%)")

    # Static-gate ablation tau, same p_bg, same selection rule
    srows = []
    for tau in TAU_STATIC_GRID:
        m = EEARF10Controller(name=STATIC, gate="static", tau_static=tau, p_bg=sel["p_bg"],
                              w_fn=w_fn, seed=seed)
        r = strip(run_prequential(m, val, y0))
        srows.append({"family": "static", "alpha": np.nan, "gamma": np.nan, "p_bg": sel["p_bg"],
                      "tau_static": tau, **r})
    dfs = pd.DataFrame(srows)
    sbest = select(dfs, max_duty_pct)
    sel["tau_static"] = float(sbest["tau_static"])
    log(f"    -> static ablation tau={sel['tau_static']} (val kT={sbest['kappa_temporal']:.4f}, "
        f"rho={sbest['duty_rho_pct']:.1f}%)")
    out = pd.concat([dfc, dfs], ignore_index=True)
    out.insert(0, "dataset", ds.name)
    return sel, out


def load_or_freeze(args, datasets, tables_dir) -> dict:
    if os.path.exists(CONFIG_PATH) and not args.retune:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        recorded = data.pop("sha256")
        actual = canonical_hash(data)
        if recorded != actual:
            raise SystemExit(f"Frozen config hash mismatch!\n recorded {recorded}\n actual   {actual}")
        missing = [k for k in datasets if k not in data["selected"]]
        if missing:
            raise SystemExit(f"Frozen config lacks datasets {missing}; rerun with --retune.")
        if data.get("quick") != args.quick:
            raise SystemExit("Frozen config was produced with a different --quick setting; use --retune.")
        log(f"Loaded frozen config {CONFIG_PATH} (SHA-256 {recorded[:16]}... verified)")
        return data

    log("STAGE 1 - validation-only tuning")
    selected, frames = {}, []
    for key, ds in datasets.items():
        sel, df = tune_dataset(key, ds, args.max_val_duty, args.seeds[0])
        selected[key] = sel
        frames.append(df)
    pd.concat(frames).to_csv(os.path.join(tables_dir, "validation_tuning.csv"), index=False)

    data_files = {
        "spain": "data/energy_dataset_clean.csv",
        "india": "data/india_electricity_real_hourly.csv",
    }
    payload = {
        "label": "EE-ARF-10 conformal gate; selected on validation splits only",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick": args.quick,
        "selection_rule": f"max validation kappa_temporal s.t. duty rho <= {args.max_val_duty}% "
                          "(ties -> fewer trees evaluated)",
        "grids": {"alpha": ALPHAS, "gamma": GAMMAS, "p_bg": P_BGS, "tau_static": TAU_STATIC_GRID},
        "fixed": {
            "n_trees": 10, "n_core": 3, "tau_init": TAU_INIT, "loss_mode": "exit_error",
            "adwin_delta": 0.002, "drift_hold": 50, "replay_buffer_ablation": 200,
            "w_fn_a_priori": W_FN_DEFAULT, "arf_params": DEFAULT_ARF_PARAMS,
            "ht_params": DEFAULT_HT_PARAMS, "tuning_seed": args.seeds[0],
        },
        "splits": {k: {"val_len": d.val_len, "test_len": d.test_len, "block": d.block_size}
                   for k, d in datasets.items()},
        "data_sha256": {k: file_sha256(p) for k, p in data_files.items() if k in datasets},
        "library_versions": lib_versions(),
        "selected": selected,
    }
    payload["sha256"] = canonical_hash(payload)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    log(f"Frozen -> {CONFIG_PATH}  SHA-256 {payload['sha256']}")
    payload.pop("sha256")
    return payload


# ---------------------------------------------------------------------- #
# Pareto utilities
# ---------------------------------------------------------------------- #
def pareto_mask(cost: np.ndarray, score: np.ndarray) -> np.ndarray:
    """True for points not dominated (lower cost, higher score)."""
    n = len(cost)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dom = (cost <= cost[i]) & (score >= score[i]) & ((cost < cost[i]) | (score > score[i]))
        if dom.any():
            keep[i] = False
    return keep


def dominance(a: pd.Series, b: pd.Series, cost: str, score="kappa_temporal") -> str:
    better_s, worse_s = a[score] > b[score], a[score] < b[score]
    cheaper, dearer = a[cost] < b[cost], a[cost] > b[cost]
    if (better_s or cheaper) and not (worse_s or dearer):
        return "dominates"
    if (worse_s or dearer) and not (better_s or cheaper):
        return "dominated"
    if not (better_s or worse_s or cheaper or dearer):
        return "equal"
    return "trade-off"


def plot_pareto(summary: pd.DataFrame, sweep: pd.DataFrame, path: str):
    import matplotlib.pyplot as plt

    dsets = list(summary["dataset"].unique())
    fig, axes = plt.subplots(len(dsets), 2, figsize=(13, 4.8 * len(dsets)), squeeze=False)
    style = {
        "Always-HT": ("#7f7f7f", "o"), "Always-HAT": ("#bcbd22", "o"),
        "Always-ARF-2": ("#9467bd", "s"), "Always-ARF-3": ("#7b2cbf", "s"),
        "Always-ARF-10": ("#d62728", "X"), STATIC: ("#ff7f0e", "^"),
        REPLAY: ("#8c564b", "v"), LITERAL: ("#17becf", "P"), PROPOSED: ("#2ca02c", "*"),
    }
    for r, ds in enumerate(dsets):
        S = summary[summary["dataset"] == ds]
        W = sweep[sweep["dataset"] == ds] if not sweep.empty else sweep
        for c, (cost, xlabel, logx) in enumerate([
            ("mean_latency_us", "Mean latency per sample (µs, predict + learn)", True),
            ("avg_trees_eval", "Average trees evaluated per sample", False),
        ]):
            ax = axes[r][c]
            if not W.empty:
                for fam, mk in [("conformal", "."), ("static", "1"), ("w_fn", "2")]:
                    F = W[W["family"] == fam]
                    if not F.empty:
                        ax.scatter(F[cost], F["kappa_temporal"], s=28, marker=mk, alpha=0.45,
                                   color="#2ca02c" if fam != "static" else "#ff7f0e",
                                   label=f"sweep: {fam}" if (r == 0 and c == 0) else None)
            pts = pd.concat([S[[cost, "kappa_temporal"]], W[[cost, "kappa_temporal"]]] if not W.empty
                            else [S[[cost, "kappa_temporal"]]])
            mask = pareto_mask(pts[cost].values, pts["kappa_temporal"].values)
            fr = pts[mask].sort_values(cost)
            ax.step(fr[cost], fr["kappa_temporal"], where="post", color="k", lw=1, ls="--",
                    label="Pareto frontier" if (r == 0 and c == 0) else None)
            for _, row in S.iterrows():
                col, mk = style.get(row["model"], ("k", "o"))
                ax.scatter(row[cost], row["kappa_temporal"], color=col, marker=mk,
                           s=240 if row["model"] == PROPOSED else 90, edgecolor="k", lw=0.8, zorder=5,
                           label=row["model"] if (r == 0 and c == 0) else None)
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Temporal kappa (vs. persistence)")
            ax.set_title(ds)
            ax.grid(alpha=0.3, which="both")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("EE-ARF-10: temporal kappa vs. compute (test split, seed-mean)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["spain", "india"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--retune", action="store_true", help="re-run validation tuning and overwrite the frozen config")
    ap.add_argument("--max-val-duty", type=float, default=35.0, help="duty constraint (%%) for validation selection")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--quick", action="store_true", help="truncate streams for a smoke test")
    ap.add_argument("--out", default="results/ee_arf")
    args = ap.parse_args()

    global CONFIG_PATH
    if args.quick:
        CONFIG_PATH = os.path.join(tempfile.gettempdir(), "ee_arf_quick_config.json")
        args.out = os.path.join(tempfile.gettempdir(), "ee_arf_quick")
        args.retune = True

    tables = os.path.join(args.out, "tables")
    figs = os.path.join(args.out, "figures")
    raw = os.path.join(args.out, "raw")
    for d in (tables, figs, raw):
        os.makedirs(d, exist_ok=True)

    t_start = time.time()
    log("=" * 88)
    log("EE-ARF-10 + DRIFT-ADAPTIVE CONFORMAL GATE - prequential experiment")
    log(f"versions: {lib_versions()}")
    log("=" * 88)

    # ---------------- Stage 0 ----------------
    log("STAGE 0 - datasets, splits, targets, class balance")
    datasets = {}
    desc_rows = []
    for key in args.datasets:
        ds = load_dataset(key)
        if args.quick:
            ds.val_len = min(ds.val_len, 1680)
            ds.test_len = min(ds.test_len, 4 * ds.block_size)
        datasets[key] = ds
        d = ds.describe()
        desc_rows.append(d)
        log(f"\n  {ds.name}")
        log(f"    target : {ds.target_definition}")
        log(f"    val    : {d['val_instances']:,} samples, positives {d['val_pos']:,} ({100*d['val_pos_rate']:.2f}%), from {d['val_start']}")
        log(f"    test   : {d['test_instances']:,} samples, positives {d['test_pos']:,} ({100*d['test_pos_rate']:.2f}%), "
            f"{d['test_start']} ({d['test_start_weekday']}) -> {d['test_end']}, {d['n_test_blocks']} weekly blocks")
        for n in ds.notes:
            log(f"    note   : {n}")
    pd.DataFrame(desc_rows).to_csv(os.path.join(tables, "dataset_summary.csv"), index=False)

    # ---------------- Stage 1 ----------------
    frozen = load_or_freeze(args, datasets, tables)

    # ---------------- Stage 2 ----------------
    log("\nSTAGE 2 - frozen models on the test split")
    per_seed_rows = []
    preds = {}  # (dataset, model) -> y_pred (first seed)
    for key, ds in datasets.items():
        cfg = frozen["selected"][key]
        test, y0 = ds.test, ds.test_prev_label
        preds[(ds.name, "__y_true__")] = np.array([y for _, y in test], dtype=np.int8)
        for si, seed in enumerate(args.seeds):
            # all models stepped in lockstep -> identical host conditions for timing
            for r in run_prequential_lockstep(main_models(cfg, seed), test, y0):
                arr = r.pop("_arrays")
                row = {"dataset": ds.name, "seed": seed, "model": r.pop("name"), **r}
                per_seed_rows.append(row)
                if si == 0:
                    preds[(ds.name, row["model"])] = arr["y_pred"]
                    np.savez_compressed(
                        os.path.join(raw, f"{key}__{row['model'].replace(' ', '_').replace('/', '-')}.npz"),
                        **arr,
                    )
                log(f"  [{ds.name} | seed {seed}] {row['model']:<44} acc={row['accuracy_pct']:6.2f}  "
                    f"kT={row['kappa_temporal']:.4f}  F1={row['macro_f1']:.4f}  FNR={row['fnr']:.4f}  "
                    f"rho={row['duty_rho_pct']:5.1f}%  trees={row['avg_trees_eval']:5.2f}  "
                    f"lat={row['mean_latency_us']:7.1f}us  p99={row['p99_latency_ms']:.3f}ms")
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(os.path.join(tables, "test_results_per_seed.csv"), index=False)

    metric_cols = ["accuracy_pct", "kappa_temporal", "kappa_cohen", "macro_f1", "pr_auc", "fnr",
                   "duty_rho_pct", "avg_trees_eval", "avg_trees_trained", "mean_latency_us",
                   "p99_latency_ms", "max_latency_ms"]
    order = [m.name for m in main_models(frozen["selected"][args.datasets[0]], 0)]
    g = per_seed.groupby(["dataset", "model"], sort=False)
    summary = g[metric_cols].mean().reset_index()
    summary["kappa_temporal_sd"] = g["kappa_temporal"].std(ddof=1).values if len(args.seeds) > 1 else 0.0
    summary["accuracy_sd"] = g["accuracy_pct"].std(ddof=1).values if len(args.seeds) > 1 else 0.0
    summary["n_seeds"] = len(args.seeds)
    summary["model"] = pd.Categorical(summary["model"], order, ordered=True)
    summary["dataset"] = pd.Categorical(summary["dataset"], [d.name for d in datasets.values()], ordered=True)
    summary = summary.sort_values(["dataset", "model"]).reset_index(drop=True)
    summary["model"] = summary["model"].astype(str)
    summary["dataset"] = summary["dataset"].astype(str)

    # ---------------- Stage 3 ----------------
    sweep = pd.DataFrame()
    if not args.no_sweep:
        log("\nSTAGE 3 - sensitivity sweeps on test (seed %d; NOT used for selection)" % args.seeds[0])
        rows = []
        for key, ds in datasets.items():
            cfg = frozen["selected"][key]
            test, y0 = ds.test, ds.test_prev_label
            jobs = []
            for a, pb in itertools.product(ALPHAS, P_BGS):
                jobs.append(("conformal", dict(alpha=a, p_bg=pb), make_proposed(
                    {**cfg, "alpha": a, "p_bg": pb}, args.seeds[0], name=f"conformal a={a} p_bg={pb}")))
            for tau in TAU_STATIC_GRID:
                jobs.append(("static", dict(tau_static=tau, p_bg=cfg["p_bg"]), EEARF10Controller(
                    name=f"static tau={tau}", gate="static", tau_static=tau, p_bg=cfg["p_bg"],
                    w_fn=cfg["w_fn"], seed=args.seeds[0])))
            for w in W_FNS:
                jobs.append(("w_fn", dict(w_fn=w), make_proposed(
                    {**cfg, "w_fn": w}, args.seeds[0], name=f"conformal w_fn={w}")))
            results = run_prequential_lockstep([m for _, _, m in jobs], test, y0)
            for (fam, params, _), r in zip(jobs, results):
                r = strip(r)
                rows.append({"dataset": ds.name, "family": fam, **params, **r})
                log(f"  [{ds.name}] {r['name']:<28} kT={r['kappa_temporal']:.4f} FNR={r['fnr']:.4f} "
                    f"rho={r['duty_rho_pct']:5.1f}% trees={r['avg_trees_eval']:.2f} lat={r['mean_latency_us']:.1f}us")
        sweep = pd.DataFrame(rows)
        sweep.to_csv(os.path.join(tables, "pareto_sweep_test.csv"), index=False)

    # ---------------- Stage 4 ----------------
    log("\nSTAGE 4 - weekly-block significance tests (seed %d predictions)" % args.seeds[0])
    test_rows, friedman_rows = [], []
    for key, ds in datasets.items():
        y_true = preds[(ds.name, "__y_true__")]
        names = [n for n in order if (ds.name, n) in preds]
        for metric in ("accuracy", "kappa_temporal"):
            blocks = {n: block_metric(y_true, preds[(ds.name, n)], ds.block_size, metric, ds.test_prev_label)
                      for n in names}
            pw = pairwise_vs_reference(blocks, PROPOSED)
            pw.insert(0, "metric", metric)
            pw.insert(0, "dataset", ds.name)
            test_rows.append(pw)
            fr = friedman(blocks)
            friedman_rows.append({"dataset": ds.name, "metric": metric, "chi2": fr["chi2"], "p": fr["p"],
                                  "n_blocks": fr.get("n_blocks"), "k_models": fr.get("k_models"),
                                  "mean_ranks(1=best)": json.dumps({k: round(v, 3) for k, v in fr["mean_ranks"].items()})})
            log(f"  [{ds.name} | {metric}] Friedman chi2={fr['chi2']:.2f} p={fr['p']:.3e}")
            for _, t in pw.iterrows():
                log(f"     {PROPOSED} vs {t['baseline']:<40} Δ={t['mean_diff']:+.4f}  W/T/L={t['wins/ties/losses']:<9} "
                    f"p_holm={t['p_holm']:.3e} {'*' if t['significant'] else ''}")
    tests_df = pd.concat(test_rows, ignore_index=True)
    tests_df.to_csv(os.path.join(tables, "weekly_block_wilcoxon_holm.csv"), index=False)
    pd.DataFrame(friedman_rows).to_csv(os.path.join(tables, "weekly_block_friedman.csv"), index=False)

    # Dominance verdicts
    verdicts = []
    for ds_name in summary["dataset"].unique():
        S = summary[summary["dataset"] == ds_name].set_index("model")
        if PROPOSED not in S.index:
            continue
        p = S.loc[PROPOSED]
        for base in ("Always-ARF-3", "Always-ARF-10"):
            b = S.loc[base]
            verdicts.append({
                "dataset": ds_name, "vs": base,
                "delta_kappa_temporal": p["kappa_temporal"] - b["kappa_temporal"],
                "latency_ratio": p["mean_latency_us"] / b["mean_latency_us"],
                "trees_ratio": p["avg_trees_eval"] / b["avg_trees_eval"],
                "on_(kT, latency)": dominance(p, b, "mean_latency_us"),
                "on_(kT, trees)": dominance(p, b, "avg_trees_eval"),
                "gap_closed_ARF3_to_ARF10_pct": np.nan,
            })
        k3, k10, kp = (S.loc["Always-ARF-3", "kappa_temporal"], S.loc["Always-ARF-10", "kappa_temporal"],
                       p["kappa_temporal"])
        verdicts[-1]["gap_closed_ARF3_to_ARF10_pct"] = 100 * (kp - k3) / (k10 - k3) if k10 != k3 else np.nan
    verdict_df = pd.DataFrame(verdicts)
    verdict_df.to_csv(os.path.join(tables, "dominance_verdicts.csv"), index=False)

    # Summary tables
    summary.to_csv(os.path.join(tables, "summary_test.csv"), index=False)
    md = ["# EE-ARF-10 test-split summary", "",
          f"Seeds: {args.seeds} (metrics are seed means). Config SHA-256: "
          f"`{canonical_hash(frozen) if 'sha256' not in frozen else frozen['sha256']}`", ""]
    for ds_name in summary["dataset"].unique():
        S = summary[summary["dataset"] == ds_name]
        md += [f"## {ds_name}", "",
               "| Model | Acc % | κ_temporal (±sd) | κ_Cohen | Macro-F1 | PR-AUC | FNR | ρ % | Trees eval | Trees trained | Mean µs | P99 ms | Max ms |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in S.iterrows():
            bold = "**" if r["model"] == PROPOSED else ""
            md.append(
                f"| {bold}{r['model']}{bold} | {r['accuracy_pct']:.2f} | {r['kappa_temporal']:.4f} (±{r['kappa_temporal_sd']:.4f}) | "
                f"{r['kappa_cohen']:.4f} | {r['macro_f1']:.4f} | {r['pr_auc']:.4f} | {r['fnr']:.4f} | "
                f"{r['duty_rho_pct']:.1f} | {r['avg_trees_eval']:.2f} | {r['avg_trees_trained']:.2f} | "
                f"{r['mean_latency_us']:.1f} | {r['p99_latency_ms']:.3f} | {r['max_latency_ms']:.3f} |")
        V = verdict_df[verdict_df["dataset"] == ds_name]
        md += ["", "Dominance (proposed vs. baseline):", ""]
        for _, v in V.iterrows():
            md.append(f"- vs {v['vs']}: Δκ_T = {v['delta_kappa_temporal']:+.4f}; latency ×{v['latency_ratio']:.2f}; "
                      f"trees ×{v['trees_ratio']:.2f} → (κ_T, latency): **{v['on_(kT, latency)']}**, "
                      f"(κ_T, trees): **{v['on_(kT, trees)']}**")
        if len(V):
            md.append(f"- Gap closed from ARF-3 to ARF-10 (κ_T): {V.iloc[-1]['gap_closed_ARF3_to_ARF10_pct']:.1f}%")
        md.append("")
    md += ["Notes: latency = one predict+learn step on this host (Python/river), not an embedded target. "
           "Avg trees evaluated ≥ 3 for EE-ARF-10 by construction (always-on core).", ""]
    with open(os.path.join(tables, "summary_test.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    plot_pareto(summary, sweep, os.path.join(figs, "pareto_kappa_vs_cost.png"))

    run_meta = {"args": vars(args), "versions": lib_versions(), "elapsed_s": round(time.time() - t_start, 1),
                "frozen_config": CONFIG_PATH}
    with open(os.path.join(args.out, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    log("\n" + "\n".join(md))
    log(f"\nDone in {time.time() - t_start:.0f}s. Outputs in {args.out}/")


if __name__ == "__main__":
    main()
