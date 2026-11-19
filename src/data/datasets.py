"""
Unified dataset registry for the EE-ARF experiments.

Every dataset returns a ``StreamDataset`` with:
  * instances      list of (x_t: dict, y_t: int)   -- causal features, binary target
  * timestamps     list of pandas.Timestamp aligned with instances
  * val_len / test_len   day-aligned split (validation first, then test)
  * block_size     samples per week (weekly-block significance tests)
  * target_definition    exact human-readable target rule (logged verbatim)

The Spain and India streams reuse the *existing* loaders unchanged
(ElectricityStreamLoader, load_real_posoco_stream) so features and targets are
bit-identical to all earlier CGT experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

SPAIN_TARGET = (
    "y_t = int(Demand_t > Demand_{t-1})   "
    "[src/data/stream_loader.py: hourly load-direction, 1 = demand rises]"
)
INDIA_TARGET = (
    "y_t = int( pct_change(Demand)_t > 0.035  OR  Demand_t >= Q85(Demand)  OR  "
    "pct_change(Net_Demand)_t > 0.05 )   "
    "[scripts/compile_real_posoco_scada.py: grid-stress event, 1 = stress]. "
    "NOTE: Q85 is computed over the FULL 2021-2023 series (non-causal label threshold)."
)
ELEC2_TARGET = (
    "y_t = int(class == 'UP')   [river.datasets.Elec2: NSW price above 24h moving average]"
)


@dataclass
class StreamDataset:
    name: str
    instances: list
    timestamps: list
    val_len: int
    test_len: int
    block_size: int
    target_definition: str
    notes: list = field(default_factory=list)

    @property
    def val(self):
        return self.instances[: self.val_len]

    @property
    def test(self):
        return self.instances[self.val_len : self.val_len + self.test_len]

    @property
    def test_prev_label(self) -> int:
        """Label immediately preceding the test split (for persistence kappa)."""
        return int(self.instances[self.val_len - 1][1])

    @property
    def val_prev_label(self) -> int:
        return int(self.val[0][1])  # no predecessor; persistence starts on itself

    def describe(self) -> dict:
        def bal(rows):
            n = len(rows)
            pos = sum(int(r[1]) for r in rows)
            return n, pos, (pos / n if n else float("nan"))

        n_v, p_v, r_v = bal(self.val)
        n_t, p_t, r_t = bal(self.test)
        ts_v0 = self.timestamps[0]
        ts_t0 = self.timestamps[self.val_len]
        ts_t1 = self.timestamps[self.val_len + self.test_len - 1]
        return {
            "dataset": self.name,
            "target_definition": self.target_definition,
            "val_instances": n_v,
            "val_pos": p_v,
            "val_pos_rate": round(r_v, 4),
            "test_instances": n_t,
            "test_pos": p_t,
            "test_pos_rate": round(r_t, 4),
            "val_start": str(ts_v0),
            "test_start": str(ts_t0),
            "test_start_weekday": pd.Timestamp(ts_t0).day_name(),
            "test_end": str(ts_t1),
            "weekly_block_size": self.block_size,
            "n_test_blocks": n_t // self.block_size,
        }


# ---------------------------------------------------------------------- #
def load_spain(csv_path: str = "data/energy_dataset_clean.csv") -> StreamDataset:
    from src.data.stream_loader import ElectricityStreamLoader

    loader = ElectricityStreamLoader(csv_path)
    rows = list(loader.iter_stream(warmup_rows=24))
    instances = [(x, int(y)) for x, y, _ in rows]
    timestamps = [ts for _, _, ts in rows]
    return StreamDataset(
        name="Spain (ENTSO-E)",
        instances=instances,
        timestamps=timestamps,
        val_len=10512,   # 438 days, ends 2016-03-14 23:00
        test_len=24528,  # 146 x 168 h
        block_size=168,
        target_definition=SPAIN_TARGET,
        notes=["Lags are positional; 4 duplicated DST fall-back hours exist in the CSV."],
    )


def load_india(csv_path: str = "data/india_electricity_real_hourly.csv") -> StreamDataset:
    from scripts.run_real_posoco_benchmark import load_real_posoco_stream

    instances = [(x, int(y)) for x, y in load_real_posoco_stream(csv_path, warmup_rows=24)]
    ts_all = pd.to_datetime(pd.read_csv(csv_path, usecols=["Timestamp"])["Timestamp"])
    timestamps = list(ts_all.iloc[24:])
    assert len(timestamps) == len(instances)
    return StreamDataset(
        name="India (POSOCO/NLDC)",
        instances=instances,
        timestamps=timestamps,
        val_len=3624,    # 151 days, ends 2022-01-30 23:00
        test_len=16800,  # 100 x 168 h, starts Monday 2022-01-31
        block_size=168,
        target_definition=INDIA_TARGET,
        notes=["Label threshold Q85 uses the full series (look-ahead in label, not features)."],
    )


def load_elec2(val_frac: float = 0.30) -> StreamDataset:
    """Public NSW Electricity benchmark (downloaded by river on first use)."""
    from river import datasets

    rows = list(datasets.Elec2())
    instances, timestamps = [], []
    origin = pd.Timestamp("1996-05-07")
    for i, (x, y) in enumerate(rows):
        x = dict(x)
        x.pop("date", None)  # normalised date index: monotone time proxy, dropped
        instances.append((x, int(y in (True, "UP", 1))))
        timestamps.append(origin + pd.Timedelta(minutes=30 * i))
    per_day, per_week = 48, 336
    n_days = len(instances) // per_day
    val_len = int(n_days * val_frac) * per_day
    test_len = ((len(instances) - val_len) // per_week) * per_week
    return StreamDataset(
        name="NSW Elec2",
        instances=instances,
        timestamps=timestamps,
        val_len=val_len,
        test_len=test_len,
        block_size=per_week,
        target_definition=ELEC2_TARGET,
        notes=["Timestamps are a synthetic 30-min index (Elec2 ships normalised dates)."],
    )


LOADERS = {"spain": load_spain, "india": load_india, "elec2": load_elec2}


def load_dataset(key: str) -> StreamDataset:
    if key not in LOADERS:
        raise KeyError(f"Unknown dataset {key!r}; choose from {sorted(LOADERS)}")
    return LOADERS[key]()
