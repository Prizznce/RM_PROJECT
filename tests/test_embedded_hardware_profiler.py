import pytest
import pandas as pd
from src.edge.embedded_hardware_profiler import EmbeddedHardwareProfiler, HARDWARE_PROFILES, MODEL_HARDWARE_SPECS


def test_hardware_targets_and_specs():
    profiler = EmbeddedHardwareProfiler()
    df = profiler.profile_all_targets()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(HARDWARE_PROFILES) * len(MODEL_HARDWARE_SPECS)
    assert "ARM Cortex-M7 (STM32H743)" in df["hardware_target"].values
    assert "Raspberry Pi CM4" in df["hardware_target"].values[0] or any("Raspberry" in x for x in df["hardware_target"].values)


def test_cortex_m7_cgt_v3_compliance():
    profiler = EmbeddedHardwareProfiler()
    df = profiler.profile_all_targets()
    m7_cgt3 = df[(df["chip_name"] == "STM32H743ZI") & (df["model_name"] == "CGT-v3 (Learned Meta-Gate)")].iloc[0]
    
    assert m7_cgt3["fits_in_memory"] is True or m7_cgt3["fits_in_memory"] == 1
    assert m7_cgt3["rt_compliant_1ms"] is True or m7_cgt3["rt_compliant_1ms"] == 1
    assert m7_cgt3["latency_us"] < 1000.0
    assert m7_cgt3["instruction_cycles_k"] < 500.0  # Under 500K cycles at 480 MHz


def test_deep_learning_mcu_memory_failure():
    profiler = EmbeddedHardwareProfiler()
    df = profiler.profile_all_targets()
    # Deep net requires 38.4 MB SRAM, which should fail on all microcontrollers
    m7_dnn = df[(df["chip_name"] == "STM32H743ZI") & (df["model_name"] == "Deep Neural Net (500 Epochs)")].iloc[0]
    m4_dnn = df[(df["chip_name"] == "STM32F407VGT6") & (df["model_name"] == "Deep Neural Net (500 Epochs)")].iloc[0]
    esp_dnn = df[(df["chip_name"] == "ESP32-S3-WROOM-1") & (df["model_name"] == "Deep Neural Net (500 Epochs)")].iloc[0]

    assert bool(m7_dnn["fits_in_memory"]) is False
    assert bool(m4_dnn["fits_in_memory"]) is False
    assert bool(esp_dnn["fits_in_memory"]) is False
    assert bool(m7_dnn["rt_compliant_1ms"]) is False


def test_battery_autonomy_multiplication():
    profiler = EmbeddedHardwareProfiler()
    df = profiler.profile_all_targets()
    m7_cgt = df[(df["chip_name"] == "STM32H743ZI") & (df["model_name"] == "CGT-v3 (Learned Meta-Gate)")].iloc[0]
    m7_arf = df[(df["chip_name"] == "STM32H743ZI") & (df["model_name"] == "Always-ARF (10 trees)")].iloc[0]

    mult = m7_cgt["battery_runtime_hrs"] / m7_arf["battery_runtime_hrs"]
    assert mult > 5.0, f"CGT battery multiplier on Cortex-M7 should exceed 5x, got {mult:.2f}x"
