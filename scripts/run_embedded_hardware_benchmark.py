"""
scripts/run_embedded_hardware_benchmark.py

Executes embedded instruction profiling across industrial microcontrollers:
  1. ARM Cortex-M7 (STM32H743)
  2. ARM Cortex-M4 (STM32F407)
  3. ARM Cortex-A72 (Raspberry Pi CM4)
  4. Xtensa LX7 (ESP32-S3)

Generates:
  results/tables/embedded_hardware_profile.csv
"""

import os
import sys
import pandas as pd

# Add repo root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.edge.embedded_hardware_profiler import EmbeddedHardwareProfiler


def main():
    print("📟 Executing Microcontroller & Embedded Hardware Profiling Suite...")
    profiler = EmbeddedHardwareProfiler(battery_wh=100.0)
    df_profile = profiler.profile_all_targets()

    out_csv = os.path.join(BASE_DIR, "results", "tables", "embedded_hardware_profile.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_profile.to_csv(out_csv, index=False)
    print(f"✅ Generated {len(df_profile)} hardware-model evaluation tuples -> {out_csv}")

    # Display Cortex-M7 Results (Primary Substation Relay Target)
    print("\n=== ARM Cortex-M7 (STM32H743 @ 480 MHz) Substation Protection Profile ===")
    m7_df = df_profile[df_profile["chip_name"] == "STM32H743ZI"]
    cols_to_print = ["model_name", "latency_us", "instruction_cycles_k", "fits_in_memory", "rt_compliant_1ms", "battery_runtime_hrs"]
    print(m7_df[cols_to_print].to_string(index=False))


if __name__ == "__main__":
    main()
