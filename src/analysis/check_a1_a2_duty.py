"""
Fast diagnostic to check true ARF duty cycles for A1, A2, and OMS-MAB.
"""
import yaml
from src.data.stream_loader import ElectricityStreamLoader
from src.controllers.a1_threshold import A1ThresholdController
from src.controllers.a2_ucb import A2UCBController
from src.controllers.oms_mab import OMSMABController

def check_duty():
    loader = ElectricityStreamLoader("data/energy_dataset_clean.csv")
    full_stream = list(loader.iter_stream(warmup_rows=24))
    val_len = 10512
    val_stream = full_stream[:val_len]
    test_stream = full_stream[val_len:]

    # A1
    a1 = A1ThresholdController(theta_hi=0.25, theta_lo=0.15, window_size=50, seed=42)
    for x, y, ts in val_stream:
        p = a1.predict_one(x)
        a1.learn_one(x, y, p)
    a1_arf_count = 0
    for x, y, ts in test_stream:
        p = a1.predict_one(x)
        if a1.active_model == "ARF":
            a1_arf_count += 1
        a1.learn_one(x, y, p)

    # A2
    a2 = A2UCBController(c=0.5, discount=0.98, seed=42)
    for x, y, ts in val_stream:
        p = a2.predict_one(x)
        a2.learn_one(x, y, p)
    a2_arf_count = 0
    for x, y, ts in test_stream:
        p = a2.predict_one(x)
        if a2.selected_arm == "ARF":
            a2_arf_count += 1
        a2.learn_one(x, y, p)

    # OMS-MAB
    oms = OMSMABController(gamma=0.98, c=0.5, seed=42)
    for x, y, ts in val_stream:
        p = oms.predict_one(x)
        oms.learn_one(x, y, p)
    oms_arf_count = 0
    for x, y, ts in test_stream:
        p = oms.predict_one(x)
        if getattr(oms, 'selected_arm', None) == "ARF":
            oms_arf_count += 1
        oms.learn_one(x, y, p)

    N = len(test_stream)
    print(f"A1 ARF Active Count: {a1_arf_count} / {N} ({a1_arf_count/N*100:.2f}%)", flush=True)
    print(f"A2 ARF Active Count: {a2_arf_count} / {N} ({a2_arf_count/N*100:.2f}%)", flush=True)
    print(f"OMS-MAB ARF Active Count: {oms_arf_count} / {N} ({oms_arf_count/N*100:.2f}%)", flush=True)

if __name__ == "__main__":
    check_duty()

