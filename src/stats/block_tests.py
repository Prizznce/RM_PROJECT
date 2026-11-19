"""
Weekly-block significance testing.

Predictions on the test stream are cut into consecutive, non-overlapping
blocks (168 h for hourly streams). Per-block scores are treated as paired
observations to soften prequential autocorrelation.

* Pairwise: two-sided Wilcoxon signed-rank (zero_method="pratt") of the
  proposed model vs every other model, Holm-Bonferroni corrected within each
  (dataset, metric) family. Paired Cohen's d_z is reported as effect size.
* Omnibus: Friedman chi-square across all compared models, with mean ranks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))  # monotone step-down
        adj[idx] = running
    return adj


def pairwise_vs_reference(blocks: dict, reference: str, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    ref = np.asarray(blocks[reference])
    for name, vals in blocks.items():
        if name == reference:
            continue
        diff = ref - np.asarray(vals)
        if np.allclose(diff, 0):
            w, p = 0.0, 1.0
        else:
            w, p = stats.wilcoxon(ref, vals, zero_method="pratt")
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        rows.append({
            "reference": reference,
            "baseline": name,
            "n_blocks": len(diff),
            "mean_diff": float(np.mean(diff)),
            "median_diff": float(np.median(diff)),
            "wins/ties/losses": f"{int((diff > 0).sum())}/{int((diff == 0).sum())}/{int((diff < 0).sum())}",
            "wilcoxon_W": float(w),
            "p_raw": float(p),
            "cohen_dz": float(np.mean(diff) / sd) if sd > 0 else 0.0,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_holm"] = holm(df["p_raw"].values)
        df["significant"] = df["p_holm"] < alpha
    return df


def friedman(blocks: dict) -> dict:
    names = list(blocks)
    mat = np.column_stack([np.asarray(blocks[n]) for n in names])
    if mat.shape[1] < 3:
        return {"chi2": float("nan"), "p": float("nan"), "mean_ranks": {}}
    chi2, p = stats.friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
    # rank 1 = best (highest score)
    ranks = stats.rankdata(-mat, axis=1).mean(axis=0)
    return {
        "chi2": float(chi2),
        "p": float(p),
        "n_blocks": int(mat.shape[0]),
        "k_models": int(mat.shape[1]),
        "mean_ranks": dict(sorted(zip(names, map(float, ranks)), key=lambda kv: kv[1])),
    }
