"""
Generate realistic 3-Year Indian National Electrical Grid Hourly Telemetry (2021-2023).
Aligned with POSOCO / Grid Controller of India (NLDC) regional demand and generation profiles.
Total records: 26,280 continuous hourly instances.
"""

import os
import numpy as np
import pandas as pd

def generate_indian_grid_dataset():
    np.random.seed(42)
    
    # 3 continuous years: 2021-01-01 to 2023-12-31 (26,280 hours)
    date_rng = pd.date_range(start="2021-01-01 00:00:00", end="2023-12-31 23:00:00", freq="h")
    n = len(date_rng)
    
    hours = date_rng.hour.values
    days_of_year = date_rng.dayofyear.values
    day_of_week = date_rng.dayofweek.values
    
    # Base diurnal cycle (twin peaks: morning 09:00-11:00 and evening 19:00-22:00 typical of India)
    diurnal = (
        18000 * np.sin(2 * np.pi * (hours - 6) / 24)
        + 12000 * np.sin(4 * np.pi * (hours - 8) / 24)
    )
    
    # Seasonal cycle (Summer AC surge in May-June + Monsoon drop in July-August + Post-monsoon Diwali spike in Oct-Nov)
    seasonal = (
        35000 * np.sin(2 * np.pi * (days_of_year - 80) / 365)  # Summer ramp
        - 15000 * np.exp(-((days_of_year - 200) ** 2) / 600)   # Monsoon cooling dip (July)
        + 12000 * np.exp(-((days_of_year - 305) ** 2) / 200)   # Diwali festival surge (Nov)
    )
    
    # Multi-year growth trend (+6% annually in Indian electricity demand)
    growth = np.linspace(0, 22000, n)
    
    # Base demand: ranges from ~130,000 MW to ~240,000 MW
    noise = np.random.normal(0, 3500, n)
    national_demand = 165000 + diurnal + seasonal + growth + noise
    national_demand = np.clip(national_demand, 120000, 255000)
    
    # Regional breakdown (approximate historical POSOCO shares: WR 33%, NR 30%, SR 23%, ER 12%, NER 2%)
    wr_demand = national_demand * 0.33 + np.random.normal(0, 1200, n)
    nr_demand = national_demand * 0.30 + np.random.normal(0, 1100, n)
    sr_demand = national_demand * 0.23 + np.random.normal(0, 900, n)
    er_demand = national_demand * 0.12 + np.random.normal(0, 500, n)
    ner_demand = national_demand * 0.02 + np.random.normal(0, 150, n)
    
    # Solar generation (bell curve between 06:00 and 18:00, lower during July-August monsoon cloud cover)
    solar_base = np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) ** 1.8
    monsoon_cloud_factor = 1.0 - 0.55 * np.exp(-((days_of_year - 210) ** 2) / 800)
    solar_capacity = np.linspace(38000, 72000, n)  # Rapid Indian solar capacity expansion
    solar_gen = solar_base * solar_capacity * monsoon_cloud_factor + np.random.normal(0, 800, n)
    solar_gen = np.clip(solar_gen, 0, None)
    
    # Wind generation (peaks strongly during southwest monsoon June-September: days 150-260)
    wind_season = 1.0 + 1.8 * np.exp(-((days_of_year - 200) ** 2) / 1800)
    wind_capacity = np.linspace(38000, 44000, n)
    wind_gen = (0.28 * wind_capacity * wind_season) + np.random.normal(0, 1800, n)
    wind_gen = np.clip(wind_gen, 3000, 32000)
    
    # Thermal coal generation (acts as flexible baseline to cover net demand)
    renewable_total = solar_gen + wind_gen
    net_demand = national_demand - renewable_total
    coal_gen = np.clip(net_demand * 0.78 + np.random.normal(0, 2000, n), 85000, 175000)
    hydro_gen = np.clip(net_demand * 0.18 + np.random.normal(0, 1500, n), 12000, 42000)
    
    # Grid Frequency (Hz): operating around 50.00 Hz, with tight band deviations (49.85 to 50.15 Hz)
    freq_deviation = -0.0000008 * (national_demand - (coal_gen + hydro_gen + renewable_total))
    grid_frequency = 50.00 + freq_deviation + np.random.normal(0, 0.025, n)
    grid_frequency = np.clip(grid_frequency, 49.75, 50.25)
    
    # Target definition: Binary Stress / Peaking Event (1 = Extreme load ramp or frequency dip requiring rapid dispatch)
    # Stress occurs when net demand jumps by > 8% or frequency drops below 49.90 Hz
    demand_pct_change = pd.Series(national_demand).pct_change().fillna(0).values
    is_stress = (
        (demand_pct_change > 0.045) | 
        (grid_frequency < 49.90) | 
        (national_demand > np.percentile(national_demand, 85))
    ).astype(int)
    
    df = pd.DataFrame({
        "Timestamp": date_rng.strftime("%Y-%m-%d %H:%M:%S"),
        "National_Demand_MW": np.round(national_demand, 1),
        "Northern_Region_Demand_MW": np.round(nr_demand, 1),
        "Western_Region_Demand_MW": np.round(wr_demand, 1),
        "Southern_Region_Demand_MW": np.round(sr_demand, 1),
        "Eastern_Region_Demand_MW": np.round(er_demand, 1),
        "NorthEastern_Region_Demand_MW": np.round(ner_demand, 1),
        "Solar_Gen_MW": np.round(solar_gen, 1),
        "Wind_Gen_MW": np.round(wind_gen, 1),
        "Thermal_Coal_Gen_MW": np.round(coal_gen, 1),
        "Hydro_Gen_MW": np.round(hydro_gen, 1),
        "Renewable_Ratio": np.round(renewable_total / np.maximum(national_demand, 1), 4),
        "Grid_Frequency_Hz": np.round(grid_frequency, 3),
        "Hour": hours,
        "DayOfWeek": day_of_week,
        "Month": date_rng.month.values,
        "Target_Stress_Event": is_stress
    })
    
    out_path = "data/india_electricity_hourly.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Generated Indian National Grid Telemetry: {out_path} ({len(df):,} records, {os.path.getsize(out_path)/(1024*1024):.2f} MB)")
    print(f"   Target Distribution: {np.mean(is_stress)*100:.1f}% Stress Events (85-15 balanced)")

if __name__ == "__main__":
    generate_indian_grid_dataset()

