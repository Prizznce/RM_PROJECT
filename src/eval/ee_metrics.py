"""
Metrics for the EE-ARF evaluation.

kappa_temporal (Zliobaite et al., 2015, "Evaluation methods and decision theory
for classification of streaming data with temporal dependence"):
    kappa_per = (p_model - p_persist) / (1 - p_persist)
where p_persist is the accuracy of the no-change classifier y_hat_t = y_{t-1}.
It is NOT Cohen's kappa (which compares with a chance classifier); both are
reported and labelled separately.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)


def persistence_predictions(y_true: np.ndarray, y_prev0: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    out = np.empty_like(y_true)
    if len(y_true) == 0:
        return out
    out[0] = y_prev0
    out[1:] = y_true[:-1]
    return out


def kappa_temporal(y_true, y_pred, y_prev0: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    p_model = float(np.mean(y_true == y_pred))
    p_per = float(np.mean(y_true == persistence_predictions(y_true, y_prev0)))
    if p_per >= 1.0:
        return 0.0
    return (p_model - p_per) / (1.0 - p_per)


def cohen_kappa(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(np.unique(np.concatenate([y_true, y_pred]))) < 2:
        return 0.0
    return float(cohen_kappa_score(y_true, y_pred))


def fnr(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    pos = y_true == 1
    if not pos.any():
        return float("nan")
    return float(np.mean(y_pred[pos] == 0))


def classification_metrics(y_true, y_pred, p1, y_prev0: int) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    p1 = np.asarray(p1, dtype=float)
    pr_auc = (
        float(average_precision_score(y_true, p1))
        if 0 < y_true.sum() < len(y_true)
        else float("nan")
    )
    return {
        "accuracy_pct": 100.0 * float(np.mean(y_true == y_pred)),
        "kappa_temporal": kappa_temporal(y_true, y_pred, y_prev0),
        "kappa_cohen": cohen_kappa(y_true, y_pred),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_1_pct": 100.0 * float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_1_pct": 100.0 * float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pr_auc": pr_auc,
        "fnr": fnr(y_true, y_pred),
    }


def cost_metrics(latency_ns, trees_eval, trees_trained, gated) -> dict:
    lat_us = np.asarray(latency_ns, dtype=float) / 1000.0
    return {
        "mean_latency_us": float(np.mean(lat_us)),
        "p50_latency_us": float(np.percentile(lat_us, 50)),
        "p95_latency_us": float(np.percentile(lat_us, 95)),
        "p99_latency_ms": float(np.percentile(lat_us, 99)) / 1000.0,
        "max_latency_ms": float(np.max(lat_us)) / 1000.0,
        "avg_trees_eval": float(np.mean(trees_eval)),
        "avg_trees_trained": float(np.mean(trees_trained)),
        "duty_rho_pct": 100.0 * float(np.mean(gated)),
    }


def block_metric(y_true, y_pred, block_size: int, metric: str, y_prev0: int) -> np.ndarray:
    """Per-block metric over consecutive, non-overlapping blocks."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_blocks = len(y_true) // block_size
    out = np.empty(n_blocks)
    for b in range(n_blocks):
        s, e = b * block_size, (b + 1) * block_size
        yt, yp = y_true[s:e], y_pred[s:e]
        if metric == "accuracy":
            out[b] = 100.0 * np.mean(yt == yp)
        elif metric == "kappa_temporal":
            prev = y_prev0 if b == 0 else int(y_true[s - 1])
            out[b] = kappa_temporal(yt, yp, prev)
        else:
            raise ValueError(metric)
    return out
