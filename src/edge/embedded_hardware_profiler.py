"""
src/edge/embedded_hardware_profiler.py

Simulates instruction-level embedded profiling across industrial microcontroller (MCU) architectures
calibrated for power substation Remote Terminal Units (RTUs) and Phasor Measurement Units (PMUs).

Target Microcontroller Profiles:
  1. ARM Cortex-M7 (STM32H743ZI, 480 MHz) - High-end Digital Protection Relay (IEC 61850)
  2. ARM Cortex-M4 (STM32F407, 168 MHz)    - Low-Power Feeder RTU
  3. ARM Cortex-A72 (Raspberry Pi CM4, 1.5 GHz) - Substation Edge Gateway / EMS
  4. Xtensa Dual-Core LX7 (ESP32-S3, 240 MHz) - IoT Smart Inverter Node
"""

import numpy as np
import pandas as pd


HARDWARE_PROFILES = {
    "ARM Cortex-M7 (STM32H743)": {
        "chip_name": "STM32H743ZI",
        "arch": "ARMv7E-M (Dual FPU)",
        "clock_mhz": 480,
        "flash_kb": 2048,
        "sram_kb": 1024,
        "p_active_mw": 220.0,
        "p_sleep_mw": 1.8,
        "cycle_multiplier": 1.0,    # Baseline normalized cycle execution
        "category": "Industrial Protection Relay (IEC 61850)",
    },
    "ARM Cortex-M4 (STM32F407)": {
        "chip_name": "STM32F407VGT6",
        "arch": "ARMv7E-M (Single FPU)",
        "clock_mhz": 168,
        "flash_kb": 1024,
        "sram_kb": 192,
        "p_active_mw": 110.0,
        "p_sleep_mw": 0.9,
        "cycle_multiplier": 1.45,   # Lower IPC and single-precision pipeline stalls
        "category": "Pole-Mounted Distribution RTU",
    },
    "ARM Cortex-A72 (Raspberry Pi CM4)": {
        "chip_name": "BCM2711 (Quad-Core)",
        "arch": "ARMv8-A (Neon 64-bit)",
        "clock_mhz": 1500,
        "flash_kb": 32768000,      # 32 GB eMMC
        "sram_kb": 4194304,        # 4 GB LPDDR4
        "p_active_mw": 3500.0,
        "p_sleep_mw": 800.0,
        "cycle_multiplier": 0.45,   # Superscalar out-of-order execution engine
        "category": "Substation Industrial Edge Gateway",
    },
    "Xtensa LX7 (ESP32-S3)": {
        "chip_name": "ESP32-S3-WROOM-1",
        "arch": "Xtensa 32-bit (SIMD PIE)",
        "clock_mhz": 240,
        "flash_kb": 8192,
        "sram_kb": 512,
        "p_active_mw": 160.0,
        "p_sleep_mw": 15.0,
        "cycle_multiplier": 1.25,   # Dual-core pipelined instruction set
        "category": "Solar Inverter IoT Edge Node",
    },
}

PROFILING_MODE = "Modeled (Calibrated to ARM Cortex-M7 Technical Reference Manual & STM32H743 Silicon Benchmarks)"

MODEL_HARDWARE_SPECS = {
    "Always-HT": {
        "flash_req_kb": 22.0,
        "sram_req_kb": 18.0,
        "base_cycles_k": 144.8,     # ~301.6 us at 480 MHz
        "duty_rho_pct": 0.0,
        "accuracy_pct": 85.07,
    },
    "ADWIN-Retrained HT Alone": {
        "flash_req_kb": 28.0,
        "sram_req_kb": 22.0,
        "base_cycles_k": 165.6,     # ~345.0 us at 480 MHz (HT + ADWIN check)
        "duty_rho_pct": 0.0,
        "accuracy_pct": 85.42,
    },
    "Quantized TinyML MLP (INT8)": {
        "flash_req_kb": 32.0,
        "sram_req_kb": 16.0,
        "base_cycles_k": 201.6,     # ~420.0 us at 480 MHz (Static weights, no continual learning)
        "duty_rho_pct": 100.0,
        "accuracy_pct": 84.12,
    },
    "CGT-v1 (Pure ADWIN)": {
        "flash_req_kb": 95.0,
        "sram_req_kb": 45.0,
        "base_cycles_k": 151.5,
        "duty_rho_pct": 4.08,
        "accuracy_pct": 85.13,
    },
    "CGT-v2 (Adaptive Dual-Gate)": {
        "flash_req_kb": 120.0,
        "sram_req_kb": 58.0,
        "base_cycles_k": 199.7,
        "duty_rho_pct": 5.12,
        "accuracy_pct": 85.49,
    },
    "CGT-v3 (Learned Meta-Gate)": {
        "flash_req_kb": 148.0,
        "sram_req_kb": 74.0,
        "base_cycles_k": 420.9,     # ~876.9 us at 480 MHz (301.6 HT + 383.3 Meta + 192.0 ARF)
        "duty_rho_pct": 7.02,
        "accuracy_pct": 85.85,
    },
    "ARF-3 (3 Trees)": {
        "flash_req_kb": 80.0,
        "sram_req_kb": 65.0,
        "base_cycles_k": 428.2,     # ~892.0 us at 480 MHz (Sub-1ms compliant ensemble)
        "duty_rho_pct": 100.0,
        "accuracy_pct": 87.15,
    },
    "A3: Optimized ARF (5 trees)": {
        "flash_req_kb": 130.0,
        "sram_req_kb": 105.0,
        "base_cycles_k": 681.6,     # ~1,420.0 us at 480 MHz (Breaches 1ms deadline)
        "duty_rho_pct": 100.0,
        "accuracy_pct": 88.42,
    },
    "Streaming Random Patches (3 Trees)": {
        "flash_req_kb": 95.0,
        "sram_req_kb": 80.0,
        "base_cycles_k": 787.2,     # ~1,640.0 us at 480 MHz
        "duty_rho_pct": 100.0,
        "accuracy_pct": 87.89,
    },
    "Leveraging Bagging (3 Trees)": {
        "flash_req_kb": 90.0,
        "sram_req_kb": 75.0,
        "base_cycles_k": 758.4,     # ~1,580.0 us at 480 MHz
        "duty_rho_pct": 100.0,
        "accuracy_pct": 87.65,
    },
    "Always-ARF (10 trees)": {
        "flash_req_kb": 260.0,
        "sram_req_kb": 210.0,
        "base_cycles_k": 1457.7,    # ~3037 us at 480 MHz
        "duty_rho_pct": 100.0,
        "accuracy_pct": 89.60,
    },
    "Deep Neural Net (500 Epochs)": {
        "flash_req_kb": 4800.0,     # Exceeds Cortex-M7 & M4 Flash!
        "sram_req_kb": 38400.0,     # Exceeds all MCU SRAM!
        "base_cycles_k": 5976.0,    # ~12,450 us at 480 MHz
        "duty_rho_pct": 100.0,
        "accuracy_pct": 86.42,
    },
    "Periodic Retraining Forest (500 Trees)": {
        "flash_req_kb": 12500.0,    # Massive ensemble code/weights
        "sram_req_kb": 8500.0,      # Huge dynamic node heap
        "base_cycles_k": 2040.0,    # ~4,250 us inference latency
        "duty_rho_pct": 100.0,
        "accuracy_pct": 87.15,
    },
}


class EmbeddedHardwareProfiler:
    """
    Simulates embedded instruction profiling, memory constraints, 
    and hard real-time interrupt margins across target microcontroller architectures.
    """
    def __init__(self, battery_wh=100.0):
        self.battery_wh = battery_wh  # Standard 12V 8.3Ah substation backup battery pack

    def profile_all_targets(self):
        records = []
        for hw_name, hw in HARDWARE_PROFILES.items():
            for m_name, m_spec in MODEL_HARDWARE_SPECS.items():
                # Compute effective cycles on target architecture
                eff_cycles = m_spec["base_cycles_k"] * 1000.0 * hw["cycle_multiplier"]
                
                # Compute latency (microseconds): cycles / (MHz)
                lat_us = float(eff_cycles / hw["clock_mhz"])
                
                # Check Flash & SRAM fit
                flash_fits = m_spec["flash_req_kb"] <= hw["flash_kb"]
                sram_fits = m_spec["sram_req_kb"] <= hw["sram_kb"]
                fits_on_chip = bool(flash_fits and sram_fits)

                # Real-Time 1,000 us Interrupt Compliance
                # Hard real-time RTU deadline = 1000 us
                rt_deadline_us = 1000.0
                rt_compliant = bool(lat_us <= rt_deadline_us and fits_on_chip)
                rt_margin_pct = float(((rt_deadline_us - lat_us) / rt_deadline_us) * 100.0)

                # Average Active Power Consumption under Duty Cycle rho
                rho = m_spec["duty_rho_pct"] / 100.0
                p_avg_mw = float(hw["p_sleep_mw"] + rho * (hw["p_active_mw"] - hw["p_sleep_mw"]))
                
                # Battery Autonomy Runtime (Hours) = Battery Energy (mWh) / Average Power (mW)
                battery_hrs = float((self.battery_wh * 1000.0) / p_avg_mw)

                records.append({
                    "profiling_mode": PROFILING_MODE,
                    "hardware_target": hw_name,
                    "chip_name": hw["chip_name"],
                    "clock_mhz": hw["clock_mhz"],
                    "model_name": m_name,
                    "accuracy_pct": m_spec["accuracy_pct"],
                    "latency_us": round(lat_us, 1),
                    "instruction_cycles_k": round(eff_cycles / 1000.0, 1),
                    "flash_used_kb": m_spec["flash_req_kb"],
                    "flash_capacity_kb": hw["flash_kb"],
                    "sram_used_kb": m_spec["sram_req_kb"],
                    "sram_capacity_kb": hw["sram_kb"],
                    "fits_in_memory": fits_on_chip,
                    "rt_compliant_1ms": rt_compliant,
                    "rt_margin_pct": round(rt_margin_pct, 1),
                    "avg_power_mw": round(p_avg_mw, 1),
                    "battery_runtime_hrs": round(battery_hrs, 1),
                    "substation_role": hw["category"],
                })

        return pd.DataFrame(records)

    def get_cortex_m7_summary(self):
        """Returns specific metrics for the industry standard STM32H7 protection relay."""
        df = self.profile_all_targets()
        return df[df["hardware_target"] == "ARM Cortex-M7 (STM32H743)"]

