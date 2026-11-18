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

SAMPLING_RATE_HZ = 50.0  # IEEE C37.118 PMU Synchrophasor Standard (50 frames/s, T = 20 ms)
SAMPLING_PERIOD_US = 20000.0  # 20 ms = 20,000 us

MODEL_HARDWARE_SPECS = {
    "Always-HT": {
        "flash_req_kb": 22.0,
        "sram_req_kb": 18.0,
        "base_cycles_k": 144.8,     # ~301.6 us at 480 MHz
        "worst_cycles_k": 212.5,    # ~442.8 us max leaf depth
        "breakdown": "301.6 HT + 0.0 Meta + 0.0 ARF",
        "duty_rho_pct": 0.0,
        "accuracy_pct": 85.07,
    },
    "ADWIN-Retrained HT Alone": {
        "flash_req_kb": 28.0,
        "sram_req_kb": 22.0,
        "base_cycles_k": 165.6,     # ~345.0 us at 480 MHz
        "worst_cycles_k": 249.8,    # ~520.4 us
        "breakdown": "301.6 HT + 43.4 ADWIN + 0.0 ARF",
        "duty_rho_pct": 0.0,
        "accuracy_pct": 85.42,
    },
    "Quantized TinyML MLP (INT8)": {
        "flash_req_kb": 32.0,
        "sram_req_kb": 16.0,
        "base_cycles_k": 201.6,     # ~420.0 us at 480 MHz
        "worst_cycles_k": 220.0,    # ~458.3 us
        "breakdown": "420.0 MLP Dense GEMM",
        "duty_rho_pct": 100.0,
        "accuracy_pct": 84.12,
    },
    "CGT-v1 (Pure ADWIN)": {
        "flash_req_kb": 95.0,
        "sram_req_kb": 45.0,
        "base_cycles_k": 151.5,     # ~315.6 us
        "worst_cycles_k": 1603.7,   # ~3,341.0 us (when escalated)
        "breakdown": "301.6 HT + 2.4 ADWIN + 11.6 ARF",
        "duty_rho_pct": 4.08,
        "accuracy_pct": 85.13,
    },
    "CGT-v2 (Adaptive Dual-Gate)": {
        "flash_req_kb": 120.0,
        "sram_req_kb": 58.0,
        "base_cycles_k": 199.7,     # ~416.0 us
        "worst_cycles_k": 833.2,    # ~1,735.8 us (when escalated)
        "breakdown": "301.6 HT + 14.2 Margin + 100.2 ARF",
        "duty_rho_pct": 5.12,
        "accuracy_pct": 85.49,
    },
    "CGT-v3 (Learned Meta-Gate)": {
        "flash_req_kb": 148.0,
        "sram_req_kb": 74.0,
        "base_cycles_k": 420.9,     # ~876.9 us at 480 MHz
        "worst_cycles_k": 1786.5,   # ~3,721.8 us (when escalated)
        "breakdown": "301.6 HT + 383.3 Meta + 192.0 ARF",
        "duty_rho_pct": 7.02,
        "accuracy_pct": 85.85,
    },
    "ARF-3 (3 Trees)": {
        "flash_req_kb": 80.0,
        "sram_req_kb": 65.0,
        "base_cycles_k": 428.2,     # ~892.1 us at 480 MHz
        "worst_cycles_k": 614.4,    # ~1,280.0 us
        "breakdown": "892.1 ARF-3 (All 3 Trees)",
        "duty_rho_pct": 100.0,
        "accuracy_pct": 87.15,
    },
    "A3: Optimized ARF (5 trees)": {
        "flash_req_kb": 130.0,
        "sram_req_kb": 105.0,
        "base_cycles_k": 681.6,     # ~1,420.0 us at 480 MHz
        "worst_cycles_k": 950.0,    # ~1,979.2 us
        "breakdown": "1420.0 ARF-5 (All 5 Trees)",
        "duty_rho_pct": 100.0,
        "accuracy_pct": 88.42,
    },
    "Streaming Random Patches (3 Trees)": {
        "flash_req_kb": 95.0,
        "sram_req_kb": 80.0,
        "base_cycles_k": 787.2,     # ~1,640.0 us at 480 MHz
        "worst_cycles_k": 1080.0,   # ~2,250.0 us
        "breakdown": "1640.0 SRP-3 Subspace Trees",
        "duty_rho_pct": 100.0,
        "accuracy_pct": 87.89,
    },
    "Leveraging Bagging (3 Trees)": {
        "flash_req_kb": 90.0,
        "sram_req_kb": 75.0,
        "base_cycles_k": 758.4,     # ~1,580.0 us at 480 MHz
        "worst_cycles_k": 1040.0,   # ~2,166.7 us
        "breakdown": "1580.0 LevBag-3 Poisson Trees",
        "duty_rho_pct": 100.0,
        "accuracy_pct": 87.65,
    },
    "Always-ARF (10 trees)": {
        "flash_req_kb": 260.0,
        "sram_req_kb": 210.0,
        "base_cycles_k": 1457.7,    # ~3036.9 us at 480 MHz
        "worst_cycles_k": 1977.6,   # ~4,120.0 us
        "breakdown": "3036.9 ARF-10 (All 10 Trees)",
        "duty_rho_pct": 100.0,
        "accuracy_pct": 89.60,
    },
    "Deep Neural Net (500 Epochs)": {
        "flash_req_kb": 4800.0,     # Exceeds Cortex-M7 & M4 Flash!
        "sram_req_kb": 38400.0,     # Exceeds all MCU SRAM!
        "base_cycles_k": 5976.0,    # ~12,450 us at 480 MHz
        "worst_cycles_k": 6816.0,   # ~14,200 us
        "breakdown": "12450.0 500-Epoch MLP Dense Layer MatMul",
        "duty_rho_pct": 100.0,
        "accuracy_pct": 86.42,
    },
    "Periodic Retraining Forest (500 Trees)": {
        "flash_req_kb": 12500.0,    # Massive ensemble code/weights
        "sram_req_kb": 8500.0,      # Huge dynamic node heap
        "base_cycles_k": 2040.0,    # ~4,250 us inference latency
        "worst_cycles_k": 2900.0,   # ~6,041.7 us
        "breakdown": "4250.0 500-Tree Batch Forest Traversal",
        "duty_rho_pct": 100.0,
        "accuracy_pct": 87.15,
    },
}


class EmbeddedHardwareProfiler:
    """
    Simulates embedded instruction profiling, memory constraints, 
    and hard real-time interrupt margins across target microcontroller architectures.
    Recomputes all battery runtime and energy metrics from an explicit sampling rate assumption:
        fs = 50.0 Hz (IEEE C37.118 PMU synchrophasor standard, T_sample = 20.0 ms)
    """
    def __init__(self, battery_wh=100.0, sampling_rate_hz=SAMPLING_RATE_HZ):
        self.battery_wh = battery_wh  # Standard 12V 8.3Ah substation backup battery pack
        self.sampling_rate_hz = sampling_rate_hz
        self.sampling_period_us = (1.0 / sampling_rate_hz) * 1e6  # e.g. 20,000 us for 50 Hz

    def profile_all_targets(self):
        records = []
        for hw_name, hw in HARDWARE_PROFILES.items():
            for m_name, m_spec in MODEL_HARDWARE_SPECS.items():
                # Compute effective cycles on target architecture
                eff_cycles = m_spec["base_cycles_k"] * 1000.0 * hw["cycle_multiplier"]
                worst_cycles = m_spec["worst_cycles_k"] * 1000.0 * hw["cycle_multiplier"]
                
                # Compute latency (microseconds): cycles / (clock MHz)
                lat_mean_us = float(eff_cycles / hw["clock_mhz"])
                lat_worst_us = float(worst_cycles / hw["clock_mhz"])
                
                # Check Flash & SRAM fit
                flash_fits = m_spec["flash_req_kb"] <= hw["flash_kb"]
                sram_fits = m_spec["sram_req_kb"] <= hw["sram_kb"]
                fits_on_chip = bool(flash_fits and sram_fits)

                # Real-Time 1,000 us Interrupt Compliance
                rt_deadline_us = 1000.0
                rt_compliant = bool(lat_mean_us <= rt_deadline_us and fits_on_chip)
                rt_margin_pct = float(((rt_deadline_us - lat_mean_us) / rt_deadline_us) * 100.0)

                # CPU Active Duty Cycle under Sampling Rate assumption:
                # Fraction of each sampling period T_sample spent in active state
                d_cpu = min(1.0, lat_mean_us / self.sampling_period_us)
                
                # Average Active Power Consumption under Sampling Period T_sample:
                # P_avg = P_sleep + D_cpu * (P_active - P_sleep)
                p_avg_mw = float(hw["p_sleep_mw"] + d_cpu * (hw["p_active_mw"] - hw["p_sleep_mw"]))
                
                # Energy consumed per sample (micro-Joules = mW * ms):
                e_sample_uj = float(
                    (hw["p_active_mw"] * (lat_mean_us / 1e6) + 
                     hw["p_sleep_mw"] * ((self.sampling_period_us - lat_mean_us) / 1e6)) * 1000.0
                )

                # Battery Autonomy Runtime on 100 Wh Substation DC Backup Pack:
                battery_hrs_100wh = float((self.battery_wh * 1000.0) / p_avg_mw)

                # Battery Autonomy Runtime on 8.25 Wh Remote Lithium Cell (2,500 mAh @ 3.3V):
                battery_hrs_cell = float(8250.0 / p_avg_mw)

                records.append({
                    "profiling_mode": PROFILING_MODE,
                    "sampling_rate_hz": self.sampling_rate_hz,
                    "sampling_period_ms": round(self.sampling_period_us / 1000.0, 1),
                    "hardware_target": hw_name,
                    "chip_name": hw["chip_name"],
                    "clock_mhz": hw["clock_mhz"],
                    "model_name": m_name,
                    "accuracy_pct": m_spec["accuracy_pct"],
                    "latency_us": round(lat_mean_us, 1),  # Backward compatibility
                    "latency_mean_us": round(lat_mean_us, 1),
                    "latency_worst_case_us": round(lat_worst_us, 1),
                    "latency_breakdown": m_spec["breakdown"],
                    "instruction_cycles_k": round(eff_cycles / 1000.0, 1),
                    "flash_used_kb": m_spec["flash_req_kb"],
                    "flash_capacity_kb": hw["flash_kb"],
                    "sram_used_kb": m_spec["sram_req_kb"],
                    "sram_capacity_kb": hw["sram_kb"],
                    "fits_in_memory": fits_on_chip,
                    "rt_compliant_1ms": rt_compliant,
                    "rt_margin_pct": round(rt_margin_pct, 1),
                    "cpu_duty_cycle_pct": round(d_cpu * 100.0, 2),
                    "energy_per_sample_uj": round(e_sample_uj, 1),
                    "avg_power_mw": round(p_avg_mw, 2),
                    "battery_runtime_hrs": round(battery_hrs_100wh, 1),  # Backward compatibility
                    "battery_runtime_100wh_hrs": round(battery_hrs_100wh, 1),
                    "battery_runtime_cell_hrs": round(battery_hrs_cell, 1),
                    "substation_role": hw["category"],
                })

        return pd.DataFrame(records)

    def get_cortex_m7_summary(self):
        """Returns specific metrics for the industry standard STM32H7 protection relay."""
        df = self.profile_all_targets()
        return df[df["hardware_target"] == "ARM Cortex-M7 (STM32H743)"]

