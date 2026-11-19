"""
Strict prequential (test-then-train) harness shared by every model in the
EE-ARF study. The label y_t is passed to the model only after its prediction
for x_t has been recorded.

Timing: one ``perf_counter_ns`` interval per step covering predict_one +
learn_one (the full per-sample compute cost), identical for all models.
"""

from __future__ import annotations

import gc
import time

import numpy as np

from src.eval.ee_metrics import classification_metrics, cost_metrics


def run_prequential_lockstep(models: list, stream, y_prev0: int) -> list[dict]:
    """Run several models over the same stream *interleaved per sample*.

    On every sample each model performs its own predict -> learn step (timed
    individually); the order of models is rotated every sample. All models
    therefore see the same host conditions (CPU frequency, background load,
    thermal state), which removes the run-order bias of sequential timing.
    Each model's predictions are identical to a standalone run.
    """
    k, n = len(models), len(stream)
    y_true = np.empty(n, dtype=np.int8)
    y_pred = np.empty((k, n), dtype=np.int8)
    p1 = np.empty((k, n), dtype=np.float64)
    lat = np.empty((k, n), dtype=np.int64)
    t_eval = np.empty((k, n), dtype=np.int16)
    t_train = np.empty((k, n), dtype=np.int32)
    gated = np.empty((k, n), dtype=bool)
    tau = np.full((k, n), np.nan)
    gloss = np.full((k, n), np.nan)
    has_tau = [hasattr(m, "last_tau") for m in models]
    has_loss = [hasattr(m, "last_gate_loss") for m in models]
    perf = time.perf_counter_ns

    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for i, (x, y) in enumerate(stream):
            y_true[i] = y
            for j in range(k):
                mi = (i + j) % k
                m = models[mi]
                t0 = perf()
                yh = m.predict_one(x)
                m.learn_one(x, y, yh)
                lat[mi, i] = perf() - t0
                y_pred[mi, i] = yh
                p1[mi, i] = m.last_p1
                t_eval[mi, i] = m.last_trees_eval
                t_train[mi, i] = m.last_trees_trained
                gated[mi, i] = m.last_gated
                if has_tau[mi]:
                    tau[mi, i] = m.last_tau
                if has_loss[mi]:
                    gloss[mi, i] = m.last_gate_loss
            if (i & 0xFF) == 0xFF:
                gc.collect()
    finally:
        if gc_was_enabled:
            gc.enable()

    out = []
    for j, m in enumerate(models):
        res = {"name": m.name}
        res.update(classification_metrics(y_true, y_pred[j], p1[j], y_prev0))
        res.update(cost_metrics(lat[j], t_eval[j], t_train[j], gated[j]))
        res.update(m.diagnostics() if hasattr(m, "diagnostics") else {})
        res["_arrays"] = {
            "y_true": y_true, "y_pred": y_pred[j], "p1": p1[j], "latency_ns": lat[j],
            "trees_eval": t_eval[j], "gated": gated[j], "tau": tau[j], "gate_loss": gloss[j],
        }
        out.append(res)
    return out


def run_prequential(model, stream, y_prev0: int) -> dict:
    n = len(stream)
    y_true = np.empty(n, dtype=np.int8)
    y_pred = np.empty(n, dtype=np.int8)
    p1 = np.empty(n, dtype=np.float64)
    lat = np.empty(n, dtype=np.int64)
    t_eval = np.empty(n, dtype=np.int16)
    t_train = np.empty(n, dtype=np.int32)
    gated = np.empty(n, dtype=bool)
    tau = np.full(n, np.nan)

    has_tau = hasattr(model, "last_tau")
    perf = time.perf_counter_ns

    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()  # avoid GC pauses being attributed to random steps
    try:
        for i, (x, y) in enumerate(stream):
            t0 = perf()
            yh = model.predict_one(x)
            # ---- prediction is final here; label revealed below ----
            model.learn_one(x, y, yh)
            lat[i] = perf() - t0

            y_true[i] = y
            y_pred[i] = yh
            p1[i] = model.last_p1
            t_eval[i] = model.last_trees_eval
            t_train[i] = model.last_trees_trained
            gated[i] = model.last_gated
            if has_tau:
                tau[i] = model.last_tau
            if (i & 0x3FF) == 0x3FF:
                gc.collect()
    finally:
        if gc_was_enabled:
            gc.enable()

    res = {"name": model.name}
    res.update(classification_metrics(y_true, y_pred, p1, y_prev0))
    res.update(cost_metrics(lat, t_eval, t_train, gated))
    res.update(model.diagnostics() if hasattr(model, "diagnostics") else {})
    res["_arrays"] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "p1": p1,
        "latency_ns": lat,
        "trees_eval": t_eval,
        "gated": gated,
        "tau": tau,
    }
    return res
