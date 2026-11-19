"""
Final report suite: Spain (ENTSO-E) and India (NLDC/POSOCO), evaluated STRICTLY
SEPARATELY under the frozen prequential protocol.

    .venv/Scripts/python scripts/run_final_report.py

* No tuning happens here. EE-ARF-10 hyper-parameters come from
  configs/ee_arf_frozen_config.json (selected on validation splits only);
  CGT-v3 / ARF-3 from configs/fair_comparison_frozen_configs.json;
  CGT-v4 from configs/cgt_v4_frozen_config.json. All three hashes are verified.
* Every model on a stream is stepped in lockstep (same host conditions for timing),
  cold-started at the first test sample, for seeds 42, 7, 123.
* Outputs: results/final_report/
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from scripts.run_ee_arf_experiment import (  # noqa: E402
    ALPHAS, CONFIG_PATH, LITERAL, P_BGS, PROPOSED, REPLAY, STATIC, TAU_STATIC_GRID,
    canonical_hash, lib_versions, make_proposed, pareto_mask,
)
from src.controllers.ee_arf_controller import AlwaysARF, AlwaysTree, EEARF10Controller  # noqa: E402
from src.controllers.legacy_adapters import CGTv3Adapter, CGTv4Adapter, OMSMABAdapter  # noqa: E402
from src.data.datasets import load_dataset  # noqa: E402
from src.eval.ee_harness import run_prequential_lockstep  # noqa: E402
from src.eval.ee_metrics import block_metric, persistence_predictions  # noqa: E402
from src.stats.block_tests import holm  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = "results/final_report"
SEEDS = [42, 7, 123]
WINDOW_30D = 720  # hourly samples
FAIR_CFG = "configs/fair_comparison_frozen_configs.json"
V4_CFG = "configs/cgt_v4_frozen_config.json"
DATA_FILES = {"spain": "data/energy_dataset_clean.csv", "india": "data/india_electricity_real_hourly.csv"}

ROLE = {
    "Always-HT": "baseline", "Always-HAT": "baseline", "Always-ARF-2": "baseline",
    "Always-ARF-3": "baseline", "Always-ARF-10": "baseline (accuracy ceiling)",
    "OMS-MAB (Wilson et al. 2024)": "base-paper baseline",
    "CGT-v3 (Learned Gate, ARF-3, Replay B=200)": "legacy (before)",
    "CGT-v4 (Amortized Replay, ARF-3)": "legacy (before)",
    STATIC: "ablation", REPLAY: "ablation", LITERAL: "ablation", PROPOSED: "proposed",
}
PRIMARY_VS = ["Always-ARF-3", "Always-HAT", STATIC, REPLAY]


def log(m=""):
    print(m, flush=True)


# ---------------------------------------------------------------------- #
# Frozen configs
# ---------------------------------------------------------------------- #
def load_frozen():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        ee = json.load(f)
    ee_hash = ee.pop("sha256")
    assert canonical_hash(ee) == ee_hash, "EE-ARF frozen config hash mismatch"

    with open(FAIR_CFG, encoding="utf-8") as f:
        fair = json.load(f)
    fair_hash = fair.pop("sha256_hash")
    assert hashlib.sha256(json.dumps(fair, indent=2, sort_keys=True).encode()).hexdigest() == fair_hash, \
        "fair-comparison config hash mismatch"

    with open(V4_CFG, encoding="utf-8") as f:
        v4 = json.load(f)
    v4_hash = v4["metadata"]["sha256"]
    chk = copy.deepcopy(v4)
    chk["metadata"]["sha256"] = "PENDING"
    assert hashlib.sha256(json.dumps(chk, indent=2, sort_keys=True).encode()).hexdigest() == v4_hash, \
        "CGT-v4 config hash mismatch"

    for k, p in DATA_FILES.items():
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert h == ee["data_sha256"][k], f"data file changed since freezing: {p}"
    return ee, ee_hash, fair, fair_hash, v4, v4_hash


def build_models(ee_cfg, fair, v4, seed):
    v3cfg = fair["best_cgt_frozen_config"]
    arfcfg = fair["best_arf3_frozen_config"]
    return [
        AlwaysTree("HT", seed=seed),
        AlwaysTree("HAT", seed=seed),
        AlwaysARF(2, seed=seed),
        AlwaysARF(3, seed=seed),
        AlwaysARF(10, seed=seed),
        OMSMABAdapter(seed=seed),
        CGTv3Adapter(v3cfg, arfcfg, seed=seed),
        CGTv4Adapter(v4["cgt_v4_amortized_arf3"], seed=seed),
        EEARF10Controller(name=STATIC, gate="static", tau_static=ee_cfg["tau_static"],
                          staleness="background", p_bg=ee_cfg["p_bg"], w_fn=ee_cfg["w_fn"], seed=seed),
        make_proposed(ee_cfg, seed, name=REPLAY, staleness="replay", p_bg=0.0, replay_buffer=200),
        make_proposed(ee_cfg, seed, name=LITERAL, loss_mode="final_error"),
        make_proposed(ee_cfg, seed),
    ]


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def rolling_mean(a, w):
    c = np.cumsum(np.insert(np.asarray(a, dtype=float), 0, 0.0))
    out = np.full(len(a), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def rolling_kappa_temporal(y_true, y_pred, y_prev0, w):
    pm = rolling_mean(y_true == y_pred, w)
    pp = rolling_mean(y_true == persistence_predictions(y_true, y_prev0), w)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pp < 1, (pm - pp) / (1 - pp), 0.0)


def risk_stats(gate_loss, alpha, w=WINDOW_30D):
    r = rolling_mean(gate_loss, w)
    r = r[~np.isnan(r)]
    return {
        "empirical_risk": float(np.mean(gate_loss)),
        "risk_minus_alpha": float(np.mean(gate_loss) - alpha),
        "pct_30d_windows_risk_le_alpha": 100.0 * float(np.mean(r <= alpha + 1e-12)),
        "max_30d_risk": float(np.max(r)),
    }


def rank_biserial(diff):
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def wilcoxon_family(blocks, ref, others, family):
    from scipy import stats
    rows = []
    for b in others:
        diff = blocks[ref] - blocks[b]
        if np.allclose(diff, 0):
            w, p = 0.0, 1.0
        else:
            w, p = stats.wilcoxon(blocks[ref], blocks[b], zero_method="pratt")
        sd = np.std(diff, ddof=1)
        rows.append({
            "family": family, "comparison": f"{ref} vs {b}", "baseline": b,
            "n_weeks": len(diff), "mean_diff": float(np.mean(diff)), "median_diff": float(np.median(diff)),
            "wins": int((diff > 1e-12).sum()), "ties": int((np.abs(diff) <= 1e-12).sum()),
            "losses": int((diff < -1e-12).sum()), "wilcoxon_W": float(w), "p_raw": float(p),
            "cohen_dz": float(np.mean(diff) / sd) if sd > 0 else 0.0,
            "rank_biserial_r": rank_biserial(diff),
        })
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_raw"].values)
    df["significant_0.05"] = df["p_holm"] < 0.05
    return df


# ---------------------------------------------------------------------- #
def main():
    t0 = time.time()
    for sub in ("", "raw"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    ee, ee_hash, fair, fair_hash, v4, v4_hash = load_frozen()
    versions = lib_versions()
    log("=" * 90)
    log("FINAL REPORT SUITE - Spain and India evaluated separately")
    log(f"EE-ARF frozen config SHA-256 {ee_hash} (verified)")
    log(f"versions {versions}")
    log("=" * 90)

    audit_rows, all_bench, all_sweep, all_stats, risk_rows = [], {}, {}, {}, []
    fig_data = {}

    for key in ("spain", "india"):
        ds = load_dataset(key)
        cfg = ee["selected"][key]
        test, y0 = ds.test, ds.test_prev_label
        y_test = np.array([y for _, y in test], dtype=int)
        y_val = np.array([y for _, y in ds.val], dtype=int)
        ts_test = pd.to_datetime(pd.Series(ds.timestamps[ds.val_len:ds.val_len + ds.test_len]))

        # ---------------- 1. audit ----------------
        raw_rows = len(pd.read_csv(DATA_FILES[key], usecols=[0]))
        val_major = int(y_val.mean() >= 0.5)
        per_acc = 100 * np.mean(y_test == persistence_predictions(y_test, y0))
        a = {
            "stream": ds.name, "total_hours_in_csv": raw_rows, "stream_instances_after_24h_warmup": len(ds.instances),
            "val_hours": ds.val_len, "val_start": str(ds.timestamps[0]), "val_end": str(ds.timestamps[ds.val_len - 1]),
            "test_hours": ds.test_len, "test_start": str(ds.timestamps[ds.val_len]),
            "test_end": str(ds.timestamps[ds.val_len + ds.test_len - 1]), "test_weeks": ds.test_len // ds.block_size,
            "unused_tail_hours": len(ds.instances) - ds.val_len - ds.test_len,
            "target_definition": ds.target_definition,
            "val_class0_pct": 100 * (1 - y_val.mean()), "val_class1_pct": 100 * y_val.mean(),
            "test_class0_pct": 100 * (1 - y_test.mean()), "test_class1_pct": 100 * y_test.mean(),
            "majority_class_from_val": val_major,
            "majority_baseline_test_acc_pct": 100 * np.mean(y_test == val_major),
            "persistence_baseline_test_acc_pct": per_acc,
        }
        audit_rows.append(a)
        log(f"\n### {ds.name}")
        for k_, v_ in a.items():
            log(f"  {k_:<36}: {v_:.2f}" if isinstance(v_, float) else f"  {k_:<36}: {v_}")

        # ---------------- 2. main benchmark ----------------
        per_seed, arrays42, preds_by_seed = [], {}, {}
        for seed in SEEDS:
            ts = time.time()
            res = run_prequential_lockstep(build_models(cfg, fair, v4, seed), test, y0)
            for r in res:
                arr = r.pop("_arrays")
                row = {"model": r.pop("name"), "seed": seed, **r}
                if row["model"] in (PROPOSED, LITERAL):
                    row.update(risk_stats(arr["gate_loss"], cfg["alpha"]))
                per_seed.append(row)
                preds_by_seed.setdefault(seed, {})[row["model"]] = arr["y_pred"].astype(int)
                if seed == SEEDS[0]:
                    arrays42[row["model"]] = arr
            log(f"  seed {seed}: {len(res)} models in lockstep, {time.time() - ts:.0f}s")
        ps = pd.DataFrame(per_seed)
        ps.to_csv(os.path.join(OUT, "raw", f"{key}_per_seed.csv"), index=False)
        for name, arr in arrays42.items():
            np.savez_compressed(os.path.join(OUT, "raw", f"{key}__{name.replace(' ', '_').replace('/', '-')}.npz"), **arr)

        order = list(dict.fromkeys(ps["model"]))
        num = ps.drop(columns=["model", "seed"]).select_dtypes("number").columns
        g = ps.groupby("model", sort=False)
        bench = g[num].mean().reindex(order)
        bench["accuracy_sd"] = g["accuracy_pct"].std(ddof=1).reindex(order)
        bench["kappa_temporal_sd"] = g["kappa_temporal"].std(ddof=1).reindex(order)
        bench = bench.reset_index()
        bench.insert(1, "role", bench["model"].map(ROLE))
        bench["fnr_pct"] = 100 * bench["fnr"]
        ref = bench.set_index("model").loc["Always-ARF-10"]
        bench["speedup_latency_vs_arf10_x"] = ref["mean_latency_us"] / bench["mean_latency_us"]
        bench["speedup_trees_vs_arf10_x"] = ref["avg_trees_eval"] / bench["avg_trees_eval"]
        bench["alpha_target"] = np.where(bench["model"].isin([PROPOSED, LITERAL]), cfg["alpha"], np.nan)
        bench["n_seeds"] = len(SEEDS)
        cols = ["model", "role", "accuracy_pct", "accuracy_sd", "kappa_temporal", "kappa_temporal_sd", "kappa_cohen",
                "macro_f1", "pr_auc", "fnr_pct", "precision_1_pct", "recall_1_pct", "avg_trees_eval",
                "avg_trees_trained", "duty_rho_pct", "mean_latency_us", "p95_latency_us", "p99_latency_ms",
                "max_latency_ms", "speedup_latency_vs_arf10_x", "speedup_trees_vs_arf10_x", "alpha_target",
                "empirical_risk", "risk_minus_alpha", "pct_30d_windows_risk_le_alpha", "max_30d_risk",
                "final_tau", "adwin_alarms", "replayed_samples", "bg_updates", "n_seeds"]
        bench = bench[[c for c in cols if c in bench.columns]]
        bench.to_csv(os.path.join(OUT, f"{key}_benchmark_results.csv"), index=False)
        all_bench[key] = bench

        # ---------------- 3. sweep ----------------
        jobs = []
        for al, pb in itertools.product(ALPHAS, P_BGS):
            jobs.append(("conformal", al, pb, np.nan, make_proposed(
                {**cfg, "alpha": al, "p_bg": pb}, SEEDS[0], name=f"conformal a={al} p_bg={pb}")))
        for tau in TAU_STATIC_GRID:
            jobs.append(("static", np.nan, cfg["p_bg"], tau, EEARF10Controller(
                name=f"static tau={tau}", gate="static", tau_static=tau, p_bg=cfg["p_bg"],
                w_fn=cfg["w_fn"], seed=SEEDS[0])))
        ts = time.time()
        sres = run_prequential_lockstep([j[-1] for j in jobs], test, y0)
        srows, sweep_arrays = [], {}
        for (fam, al, pb, tau, _), r in zip(jobs, sres):
            arr = r.pop("_arrays")
            sweep_arrays[r["name"]] = arr
            row = {"family": fam, "alpha": al, "p_bg": pb, "tau_static": tau, "gamma": cfg["gamma"],
                   "w_fn": cfg["w_fn"], "is_frozen_selection": fam == "conformal" and al == cfg["alpha"] and pb == cfg["p_bg"],
                   **{k_: v_ for k_, v_ in r.items()}}
            row["fnr_pct"] = 100 * row.pop("fnr")
            if fam == "conformal":
                row.update(risk_stats(arr["gate_loss"], al))
                risk_rows.append({"stream": ds.name, "alpha": al, "p_bg": pb, **risk_stats(arr["gate_loss"], al),
                                  "duty_rho_pct": row["duty_rho_pct"], "kappa_temporal": row["kappa_temporal"]})
            srows.append(row)
        sweep = pd.DataFrame(srows)
        sweep.to_csv(os.path.join(OUT, f"{key}_pareto_sweep.csv"), index=False)
        all_sweep[key] = sweep
        log(f"  sweep: {len(jobs)} configs in lockstep, {time.time() - ts:.0f}s")

        # ---------------- 4. statistics (weekly blocks, seed-mean per block) ----------------
        frames = []
        for metric in ("accuracy", "kappa_temporal"):
            B = {name: np.mean([block_metric(y_test, preds_by_seed[s][name], ds.block_size, metric, y0)
                                for s in SEEDS], axis=0) for name in order}
            prim = wilcoxon_family(B, PROPOSED, PRIMARY_VS, "primary (Holm m=4)")
            sec_names = [n for n in order if n not in PRIMARY_VS + [PROPOSED]]
            sec = wilcoxon_family(B, PROPOSED, sec_names, f"secondary (Holm m={len(sec_names)})")
            f = pd.concat([prim, sec], ignore_index=True)
            f.insert(0, "metric", metric)
            f.insert(0, "stream", ds.name)
            frames.append(f)
        st = pd.concat(frames, ignore_index=True)
        st.to_csv(os.path.join(OUT, f"statistical_tests_{key}.csv"), index=False)
        all_stats[key] = st

        fig_data[key] = {"ds": ds, "ts": ts_test, "y": y_test, "y0": y0, "arr": arrays42,
                         "sweep_arr": sweep_arrays, "cfg": cfg, "bench": bench, "sweep": sweep}
        del preds_by_seed

    pd.DataFrame(audit_rows).to_csv(os.path.join(OUT, "dataset_audit.csv"), index=False)
    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(os.path.join(OUT, "conformal_risk_verification.csv"), index=False)

    # ---------------- 5. figures ----------------
    make_figures(fig_data)

    # ---------------- 6. NaN / sanity verification ----------------
    problems = verify_outputs(all_bench, all_sweep, all_stats, audit_rows)

    # ---------------- 7. report ----------------
    md = write_report(audit_rows, all_bench, all_sweep, all_stats, risk_df, ee, ee_hash, fair_hash, v4_hash,
                      versions, problems, time.time() - t0)
    log("\n" + md)
    if problems:
        raise SystemExit("VERIFICATION FAILED:\n" + "\n".join(problems))
    log(f"\nAll outputs verified (no NaN in mandatory fields). Total {time.time() - t0:.0f}s -> {OUT}/")


# ---------------------------------------------------------------------- #
# Verification
# ---------------------------------------------------------------------- #
MANDATORY = ["accuracy_pct", "kappa_temporal", "kappa_cohen", "macro_f1", "pr_auc", "fnr_pct", "precision_1_pct",
             "recall_1_pct", "avg_trees_eval", "duty_rho_pct", "mean_latency_us", "p95_latency_us",
             "p99_latency_ms", "max_latency_ms", "speedup_latency_vs_arf10_x"]


def verify_outputs(all_bench, all_sweep, all_stats, audit_rows):
    problems = []
    for key, b in all_bench.items():
        for c in MANDATORY:
            if b[c].isna().any() or np.isinf(b[c]).any():
                problems.append(f"{key} benchmark: NaN/inf in {c} for {list(b.loc[b[c].isna(), 'model'])}")
        for c in ("empirical_risk", "pct_30d_windows_risk_le_alpha"):
            if b.loc[b["model"] == PROPOSED, c].isna().any():
                problems.append(f"{key}: missing {c} for proposed")
        if len(b) != 12:
            problems.append(f"{key}: expected 12 models, got {len(b)}")
        if not (b["accuracy_pct"].between(0, 100).all() and b["duty_rho_pct"].between(0, 100).all()):
            problems.append(f"{key}: out-of-range accuracy/duty")
        s = all_sweep[key]
        for c in ["kappa_temporal", "avg_trees_eval", "mean_latency_us", "duty_rho_pct"]:
            if s[c].isna().any():
                problems.append(f"{key} sweep: NaN in {c}")
        if s.loc[s["family"] == "conformal", "empirical_risk"].isna().any():
            problems.append(f"{key} sweep: NaN empirical risk")
        st = all_stats[key]
        for c in ["p_raw", "p_holm", "mean_diff", "cohen_dz", "rank_biserial_r"]:
            if st[c].isna().any():
                problems.append(f"{key} stats: NaN in {c}")
        if (st["wins"] + st["ties"] + st["losses"] != st["n_weeks"]).any():
            problems.append(f"{key} stats: W/T/L do not sum to n_weeks")
    for a in audit_rows:
        if a["val_hours"] % 24 or a["test_hours"] % 168:
            problems.append(f"{a['stream']}: split not day/week aligned")
        if not a["test_start"].endswith("00:00:00"):
            problems.append(f"{a['stream']}: test split does not start at midnight")
    return problems


# ---------------------------------------------------------------------- #
# Figures
# ---------------------------------------------------------------------- #
STYLE = {
    "Always-HT": ("#7f7f7f", "o"), "Always-HAT": ("#bcbd22", "o"), "Always-ARF-2": ("#c5b0d5", "s"),
    "Always-ARF-3": ("#7b2cbf", "s"), "Always-ARF-10": ("#d62728", "X"),
    "OMS-MAB (Wilson et al. 2024)": ("#1f77b4", "D"),
    "CGT-v3 (Learned Gate, ARF-3, Replay B=200)": ("#8c8c8c", "h"),
    "CGT-v4 (Amortized Replay, ARF-3)": ("#e377c2", "h"),
    STATIC: ("#ff7f0e", "^"), REPLAY: ("#8c564b", "v"), LITERAL: ("#17becf", "P"), PROPOSED: ("#2ca02c", "*"),
}


def _pareto_panel(ax, bench, sweep, cost, logx, legend):
    for fam, col, lab in [("conformal", "#2ca02c", "EE-ARF-10 conformal sweep (α × p_bg) frontier"),
                          ("static", "#ff7f0e", "EE-ARF-10 static-τ sweep")]:
        F = sweep[sweep["family"] == fam]
        ax.scatter(F[cost], F["kappa_temporal"], s=18, color=col, alpha=0.35, zorder=2)
        m = pareto_mask(F[cost].values, F["kappa_temporal"].values)
        fr = F[m].sort_values(cost)
        ax.plot(fr[cost], fr["kappa_temporal"], color=col, lw=1.6, alpha=0.8, zorder=3,
                ls="-" if fam == "conformal" else ":", label=lab if legend else None)
    for _, r in bench.iterrows():
        col, mk = STYLE.get(r["model"], ("k", "o"))
        ax.scatter(r[cost], r["kappa_temporal"], color=col, marker=mk, edgecolor="k", lw=0.7, zorder=5,
                   s=300 if r["model"] == PROPOSED else 90, label=r["model"] if legend else None)
    if logx:
        ax.set_xscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylabel("Temporal kappa (vs. persistence)")


def make_figures(fig_data):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    keys = list(fig_data)
    for fname, cost, xlabel, logx in [
        ("fig1_pareto_kappa_vs_latency.png", "mean_latency_us", "Mean latency per sample (µs, predict+learn, host)", True),
        ("fig2_pareto_kappa_vs_trees.png", "avg_trees_eval", "Average trees evaluated per sample", False),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
        for i, k in enumerate(keys):
            d = fig_data[k]
            _pareto_panel(axes[i], d["bench"], d["sweep"], cost, logx, legend=(i == 0))
            axes[i].set_title(d["ds"].name, fontweight="bold")
            axes[i].set_xlabel(xlabel)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.suptitle("Temporal kappa vs. " + ("latency" if logx else "trees evaluated")
                     + " — test split, mean of 3 seeds (sweep: seed 42)", fontweight="bold")
        fig.tight_layout(rect=(0, 0.13, 1, 0.96))
        fig.savefig(os.path.join(OUT, fname), dpi=300)
        plt.close(fig)

    # fig3 staleness
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex="col")
    for i, k in enumerate(keys):
        d = fig_data[k]
        cfg = d["cfg"]
        series = [("Always-ARF-10", d["arr"]["Always-ARF-10"], "#d62728", "-"),
                  (REPLAY, d["arr"][REPLAY], "#8c564b", "-"),
                  (f"{PROPOSED} [frozen p_bg={cfg['p_bg']}]", d["arr"][PROPOSED], "#2ca02c", "-")]
        alt = f"conformal a={cfg['alpha']} p_bg=0.2"
        if cfg["p_bg"] != 0.2 and alt in d["sweep_arr"]:
            series.append((f"EE-ARF-10 conformal, p_bg=0.20 (same α, γ)", d["sweep_arr"][alt], "#98df8a", "--"))
        for name, arr, col, ls in series:
            yp = arr["y_pred"].astype(int)
            axes[0][i].plot(d["ts"], rolling_kappa_temporal(d["y"], yp, d["y0"], WINDOW_30D), color=col, ls=ls, lw=1.2, label=name)
            axes[1][i].plot(d["ts"], 100 * rolling_mean(d["y"] == yp, WINDOW_30D), color=col, ls=ls, lw=1.2, label=name)
        axes[0][i].set_title(d["ds"].name, fontweight="bold")
        axes[0][i].set_ylabel("Rolling 30-day temporal kappa")
        axes[1][i].set_ylabel("Rolling 30-day accuracy (%)")
        for ax in (axes[0][i], axes[1][i]):
            ax.grid(alpha=0.3)
        axes[1][i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[0][i].legend(fontsize=8, loc="lower left")
    fig.suptitle("Staleness ablation: dormant replay (B=200) vs. subsampled background updates vs. Always-ARF-10 (seed 42)",
                 fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "fig3_staleness_ablation.png"), dpi=300)
    plt.close(fig)

    # fig4 conformal risk tracking
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for i, k in enumerate(keys):
        d = fig_data[k]
        arr, al = d["arr"][PROPOSED], d["cfg"]["alpha"]
        ax = axes[i]
        ax.plot(d["ts"], rolling_mean(arr["gate_loss"], WINDOW_30D), color="#2ca02c", lw=1.4,
                label=r"rolling 30-day empirical risk $\hat{\ell}_t$")
        ax.plot(d["ts"], rolling_mean(arr["gate_loss"], 168), color="#2ca02c", lw=0.6, alpha=0.4,
                label=r"rolling 7-day $\hat{\ell}_t$")
        ax.axhline(al, color="k", ls="--", lw=1.2, label=f"target α = {al}")
        ax.set_ylabel("Exit-error risk (w_FN-weighted)")
        ax2 = ax.twinx()
        ax2.plot(d["ts"], arr["tau"], color="#9467bd", lw=0.7, alpha=0.7, label=r"threshold $\tau_t$")
        ax2.set_ylabel(r"$\tau_t$", color="#9467bd")
        ax2.set_ylim(0, 1)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
        ax.set_title(f"{d['ds'].name} — γ={d['cfg']['gamma']}, w_FN={d['cfg']['w_fn']}, mean ℓ̂="
                     f"{np.mean(arr['gate_loss']):.4f}", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("Conformal risk tracking of EE-ARF-10 (proposed, seed 42)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "fig4_conformal_risk_tracking.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------- #
# Report
# ---------------------------------------------------------------------- #
def _f(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


def _short(name):
    return {"OMS-MAB (Wilson et al. 2024)": "OMS-MAB (Wilson et al. 2024)",
            "CGT-v3 (Learned Gate, ARF-3, Replay B=200)": "CGT-v3 (learned gate, replay B=200)",
            "CGT-v4 (Amortized Replay, ARF-3)": "CGT-v4 (amortized replay)",
            STATIC: "EE-ARF-10 · static margin gate", REPLAY: "EE-ARF-10 · dormant replay B=200",
            LITERAL: "EE-ARF-10 · conformal, final-error loss",
            PROPOSED: "**EE-ARF-10 · conformal + subsampled BG (proposed)**"}.get(name, name)


def write_report(audit, all_bench, all_sweep, all_stats, risk_df, ee, ee_hash, fair_hash, v4_hash, versions,
                 problems, elapsed):
    L = ["# Final Report Summary — EE-ARF-10 with Drift-Adaptive Conformal Gate", "",
         "Spain (ENTSO-E) and India (NLDC/POSOCO) are evaluated **strictly separately**: separate streams, "
         "separate validation-selected configs, separate statistics. Protocol: strict prequential "
         "test-then-train, cold start at the first test hour, day-aligned splits, seeds 42 / 7 / 123 "
         "(tables = seed means), all models stepped in lockstep so they share host timing conditions.", "",
         "## Reproducibility", "",
         f"- EE-ARF-10 frozen config `configs/ee_arf_frozen_config.json` — SHA-256 `{ee_hash}` (verified; "
         "canonical-JSON hash, line-ending independent)",
         f"- CGT-v3 / ARF-3 config `configs/fair_comparison_frozen_configs.json` — SHA-256 `{fair_hash}` (verified)",
         f"- CGT-v4 config `configs/cgt_v4_frozen_config.json` — SHA-256 `{v4_hash}` (verified)",
         f"- Data files SHA-256 match the values recorded at freezing time. Library versions: "
         + ", ".join(f"{k} {v}" for k, v in versions.items()), "",
         "## 1. Dataset & target audit", "",
         "| | " + " | ".join(a["stream"] for a in audit) + " |", "|---|" + "---|" * len(audit)]
    rows = [("Total hours in CSV", "total_hours_in_csv", "{:,}"),
            ("Stream instances (after 24 h lag warm-up)", "stream_instances_after_24h_warmup", "{:,}"),
            ("Validation hours", "val_hours", "{:,}"), ("Validation range", None, None),
            ("Frozen test hours", "test_hours", "{:,}"), ("Test range", None, None),
            ("Test weeks (168 h blocks)", "test_weeks", "{}"),
            ("Validation class 0 / 1 (%)", None, None), ("Test class 0 / 1 (%)", None, None),
            ("Majority class (from validation)", "majority_class_from_val", "{}"),
            ("Majority-class baseline test accuracy (%)", "majority_baseline_test_acc_pct", "{:.2f}"),
            ("Persistence (ŷ_t = y_{t−1}) test accuracy (%)", "persistence_baseline_test_acc_pct", "{:.2f}")]
    for lab, k, fmt in rows:
        if lab == "Validation range":
            vals = [f"{a['val_start'][:10]} → {a['val_end'][:16]}" for a in audit]
        elif lab == "Test range":
            vals = [f"{a['test_start'][:10]} → {a['test_end'][:16]}" for a in audit]
        elif lab.startswith("Validation class"):
            vals = [f"{a['val_class0_pct']:.2f} / {a['val_class1_pct']:.2f}" for a in audit]
        elif lab.startswith("Test class"):
            vals = [f"{a['test_class0_pct']:.2f} / {a['test_class1_pct']:.2f}" for a in audit]
        else:
            vals = [fmt.format(a[k]) for a in audit]
        L.append(f"| {lab} | " + " | ".join(vals) + " |")
    L += ["", "**Binary targets (exact):**", ""]
    for a in audit:
        L.append(f"- **{a['stream']}**: `{a['target_definition']}`")
    L += ["", "> India: the 85th-percentile demand threshold in the label is computed over the full 2021–2023 "
          "series (look-ahead in the *label definition*, not in the features). Class-1 share rises from "
          f"{audit[1]['val_class1_pct']:.1f}% (validation) to {audit[1]['test_class1_pct']:.1f}% (test).", ""]

    hdr = ("| Model | Acc % | κ_temporal | κ_Cohen | Macro-F1 | PR-AUC | FNR % | Prec₁ % | Rec₁ % | Trees/sample | ρ % "
           "| Mean µs | P95 µs | P99 ms | Max ms | Speed-up vs ARF-10 (latency / trees) |")
    sep = "|---|" + "---:|" * 15
    for idx, key in enumerate(all_bench):
        b = all_bench[key]
        name = audit[idx]["stream"]
        L += [f"## {2 + idx}. {name} — benchmark (test split, {audit[idx]['test_weeks']} weeks)", "", hdr, sep]
        for _, r in b.iterrows():
            L.append(
                f"| {_short(r['model'])} | {_f(r['accuracy_pct'])} ± {_f(r['accuracy_sd'])} | "
                f"{_f(r['kappa_temporal'], 4)} ± {_f(r['kappa_temporal_sd'], 4)} | {_f(r['kappa_cohen'], 4)} | "
                f"{_f(r['macro_f1'], 4)} | {_f(r['pr_auc'], 4)} | {_f(r['fnr_pct'])} | {_f(r['precision_1_pct'])} | "
                f"{_f(r['recall_1_pct'])} | {_f(r['avg_trees_eval'])} | {_f(r['duty_rho_pct'], 1)} | "
                f"{_f(r['mean_latency_us'], 1)} | {_f(r['p95_latency_us'], 1)} | {_f(r['p99_latency_ms'], 3)} | "
                f"{_f(r['max_latency_ms'], 3)} | {_f(r['speedup_latency_vs_arf10_x'])}× / {_f(r['speedup_trees_vs_arf10_x'])}× |")
        L.append("")

    # Ablation study
    L += ["## 4. Ablation study (both streams)", "",
          "| Stream | Variant | What changes vs. proposed | κ_temporal | Acc % | FNR % | Trees/sample | ρ % | Mean µs | Max ms |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    what = {PROPOSED: "—", STATIC: "fixed τ (validation-tuned), no adaptation",
            REPLAY: "no background updates; trees 4–10 learn only when gated + B=200 replay on ADWIN alarm",
            LITERAL: "conformal loss = final 0-1 error instead of exit error",
            "CGT-v4 (Amortized Replay, ARF-3)": "previous architecture (v4)",
            "CGT-v3 (Learned Gate, ARF-3, Replay B=200)": "previous architecture (v3)",
            "Always-ARF-10": "no early exit (all 10 trees always)"}
    for idx, key in enumerate(all_bench):
        b = all_bench[key].set_index("model")
        for m in [PROPOSED, STATIC, REPLAY, LITERAL, "CGT-v4 (Amortized Replay, ARF-3)",
                  "CGT-v3 (Learned Gate, ARF-3, Replay B=200)", "Always-ARF-10"]:
            r = b.loc[m]
            L.append(f"| {audit[idx]['stream']} | {_short(m)} | {what[m]} | {_f(r['kappa_temporal'], 4)} | "
                     f"{_f(r['accuracy_pct'])} | {_f(r['fnr_pct'])} | {_f(r['avg_trees_eval'])} | "
                     f"{_f(r['duty_rho_pct'], 1)} | {_f(r['mean_latency_us'], 1)} | {_f(r['max_latency_ms'], 3)} |")
    L.append("")

    # Conformal risk
    L += ["## 5. Conformal risk verification (target α vs. empirical risk ℓ̂)", "",
          "ℓ̂ = mean exit-error loss (w_FN-weighted, normalised). Online conformal control guarantees the "
          "**long-run average** ℓ̂ → α; it does not guarantee ℓ̂ ≤ α in every window. Both are reported.", "",
          "| Stream | α | p_bg | ℓ̂ (overall) | ℓ̂ − α | % of 30-day windows with ℓ̂ ≤ α | max 30-day ℓ̂ | ρ % | κ_temporal |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in risk_df.iterrows():
        sel = ""
        key = "spain" if r["stream"].startswith("Spain") else "india"
        c = ee["selected"][key]
        if r["alpha"] == c["alpha"] and r["p_bg"] == c["p_bg"]:
            sel = " ◀ frozen"
        L.append(f"| {r['stream']} | {r['alpha']}{sel} | {r['p_bg']} | {_f(r['empirical_risk'], 4)} | "
                 f"{r['risk_minus_alpha']:+.4f} | {_f(r['pct_30d_windows_risk_le_alpha'], 1)} | "
                 f"{_f(r['max_30d_risk'], 4)} | {_f(r['duty_rho_pct'], 1)} | {_f(r['kappa_temporal'], 4)} |")
    L.append("")
    for idx, key in enumerate(all_bench):
        b = all_bench[key].set_index("model")
        r = b.loc[LITERAL]
        L.append(f"- {audit[idx]['stream']}, final-error-loss ablation (α={ee['selected'][key]['alpha']}): "
                 f"ℓ̂ = {_f(r['empirical_risk'], 4)} ⇒ target unreachable, τ saturates (ρ = {_f(r['duty_rho_pct'], 1)}%).")
    L.append("")

    # Stats
    L += ["## 6. Statistical significance (weekly blocks, seed-mean scores, two-sided Wilcoxon signed-rank, Holm-Bonferroni)", "",
          "Primary family = the four pre-specified comparisons (Holm m=4 per stream × metric). "
          "Secondary family = all remaining models (separate Holm correction, exploratory).", "",
          "| Stream | Metric | Family | Proposed vs. | Mean Δ | W / T / L (weeks) | W | p (raw) | p (Holm) | Cohen d_z | rank-biserial r | Sig. |",
          "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|:---:|"]
    for key, st in all_stats.items():
        for _, r in st.iterrows():
            fam = "primary" if r["family"].startswith("primary") else "secondary"
            L.append(f"| {r['stream']} | {r['metric']} | {fam} | {_short(r['baseline']).replace('**', '')} | "
                     f"{r['mean_diff']:+.4f} | {r['wins']} / {r['ties']} / {r['losses']} | {r['wilcoxon_W']:.1f} | "
                     f"{r['p_raw']:.3e} | {r['p_holm']:.3e} | {r['cohen_dz']:+.3f} | {r['rank_biserial_r']:+.3f} | "
                     f"{'✔' if r['significant_0.05'] else '✘'} |")
    L += ["", "## Notes", "",
          "- **κ_temporal** (Žliobaitė et al., 2015) compares against the persistence classifier ŷ_t = y_{t−1}; "
          "**κ_Cohen** compares against chance. They are different metrics and are reported separately.",
          "- **Trees/sample** counts trees evaluated on the deployed decision path. CGT-v3 additionally "
          "shadow-evaluates its ARF-3 on every step to train its gate (CGT-v4: only when HT is wrong); these "
          "shadow evaluations are not included in the column but are included in measured latency.",
          "- **ρ** = % of samples invoking the heavy tier (trees 4–10 for EE-ARF-10 / ARF-10; the ARF arm for "
          "OMS-MAB and CGT).",
          "- OMS-MAB uses the repository implementation (D-UCB, HDDM_A, HT vs ARF-10 with the repo's "
          "ARFFullWrapper settings: grace_period=50, δ=0.01). All other ARF variants share grace_period=200, δ=0.001.",
          "- Latency is measured on the host (Python 3.13 + river 0.27, one predict+learn step); no embedded or "
          "battery extrapolation is made.",
          f"- Verification: {'PASSED — no NaN/inf in mandatory fields, splits day/week-aligned, W/T/L consistent.' if not problems else 'FAILED: ' + '; '.join(problems)}",
          f"- Wall-clock for the full suite: {elapsed / 60:.1f} min."]
    md = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "REPORT_SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return md


if __name__ == "__main__":
    main()
