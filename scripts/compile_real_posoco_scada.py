"""
Compile Real-World Indian National Power Grid (POSOCO / NLDC) SCADA Telemetry.
Aggregates 28 monthly OSI Monarch SCADA workbooks (September 2021 - December 2023).
Processes 5-minute (2021-2022) and 10-second (2022-2023) telemetry into a clean,
continuous hourly streaming dataset: data/india_electricity_real_hourly.csv.
"""

import os
import sys
import glob
import re
import time
from collections import defaultdict
from datetime import datetime
import multiprocessing as mp
import numpy as np
import pandas as pd
import openpyxl

MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11,
    'novemeber': 11, 'december': 12
}

def parse_month_files(folder):
    files = glob.glob(os.path.join(folder, "*.xlsx"))
    parsed = []
    for f in files:
        fname = os.path.basename(f)
        if "June 2025" in fname or "~$" in fname:
            continue
        m = re.match(r'([A-Za-z]+)\s+(\d{4})\.xlsx', fname)
        if m:
            m_name = m.group(1).lower()
            year = int(m.group(2))
            month = MONTH_MAP.get(m_name)
            if month:
                parsed.append((year, month, f, fname))
    parsed.sort(key=lambda x: (x[0], x[1]))
    return parsed

def safe_float(v):
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except:
        return 0.0

def process_single_workbook(item):
    year, month, fpath, fname = item
    t0 = time.time()
    try:
        wb = openpyxl.load_workbook(fpath, read_only=True)
        if "Sheet1" not in wb.sheetnames:
            return None
        sheet = wb["Sheet1"]
        
        # Hourly accumulator: hour_dt -> [thm, hyd, gas, npc, wind, sol, dem, ine, count]
        hourly = defaultdict(lambda: [0.0] * 8 + [0])
        
        row_count = 0
        for i, row in enumerate(sheet.iter_rows(values_only=True)):
            if i < 2:
                continue
            t = row[0]
            if not isinstance(t, datetime):
                continue
            row_count += 1
            h_key = t.replace(minute=0, second=0, microsecond=0)
            acc = hourly[h_key]
            
            acc[0] += safe_float(row[1])   # TOTAL_THM_ONLY|P
            acc[1] += safe_float(row[2])   # TOTAL_NLDC_HYD|P
            acc[2] += safe_float(row[3])   # TOTAL_NLDC_GAS|P
            acc[3] += safe_float(row[4])   # TOTAL_NLDC_NPC|P
            acc[4] += safe_float(row[5])   # ALL_INDIA_WIND|P
            acc[5] += safe_float(row[6])   # ALL_IND_SOLAR|P
            acc[6] += safe_float(row[10])  # NLDC_DEMAND|P
            acc[7] += safe_float(row[11])  # INTRA_SYSTEM_INTERTIA|P
            acc[8] += 1
            
        elapsed = time.time() - t0
        print(f"[{year}-{month:02d}] {fname}: {row_count:,} rows -> {len(hourly)} hourly intervals in {elapsed:.1f}s", flush=True)
        
        # Convert to list of dicts
        records = []
        for h_dt, vals in hourly.items():
            cnt = vals[8]
            if cnt == 0:
                continue
            records.append({
                "Timestamp": h_dt,
                "Thermal_Coal_MW": vals[0] / cnt,
                "Hydro_MW": vals[1] / cnt,
                "Gas_MW": vals[2] / cnt,
                "Nuclear_MW": vals[3] / cnt,
                "Wind_MW": vals[4] / cnt,
                "Solar_MW": vals[5] / cnt,
                "Demand_MW": vals[6] / cnt,
                "System_Inertia": vals[7] / cnt,
                "Raw_Samples_Per_Hour": cnt
            })
        return records
    except Exception as e:
        print(f"Error reading {fname}: {e}", flush=True)
        return None

def main():
    folder = "Electricity Demand, Solar and Wind Generation Data/Electricity Demand, Solar and Wind Generation Data"
    items = parse_month_files(folder)
    print(f"Discovered {len(items)} monthly SCADA workbooks from {items[0][0]}-{items[0][1]:02d} to {items[-1][0]}-{items[-1][1]:02d}.")
    
    # Process files in parallel
    num_workers = min(4, mp.cpu_count())
    print(f"Compiling real-world SCADA streams using {num_workers} parallel workers...")
    
    all_records = []
    t_start = time.time()
    with mp.Pool(num_workers) as pool:
        results = pool.map(process_single_workbook, items)
        
    for res in results:
        if res:
            all_records.extend(res)
            
    print(f"\nAll files processed in {time.time() - t_start:.1f}s. Assembling master time series...")
    df = pd.DataFrame(all_records)
    df = df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"]).reset_index(drop=True)
    
    # Fill any isolated missing timestamps via hourly reindexing
    min_date = df["Timestamp"].min()
    max_date = df["Timestamp"].max()
    full_idx = pd.date_range(start=min_date, end=max_date, freq="h", name="Timestamp")
    print(f"Continuous time range: {min_date} to {max_date} ({len(full_idx)} total hours)")
    
    df = df.set_index("Timestamp").reindex(full_idx)
    # Forward-fill minor gaps (e.g. 1-2 missing hours)
    df = df.interpolate(method="time").bfill()
    df = df.reset_index()
    
    # Calculate physical grid metrics
    df["Renewable_Total_MW"] = np.maximum(0, df["Wind_MW"] + df["Solar_MW"])
    df["Total_Generation_MW"] = (
        df["Thermal_Coal_MW"] + df["Hydro_MW"] + df["Gas_MW"] + 
        df["Nuclear_MW"] + df["Renewable_Total_MW"]
    )
    df["Renewable_Ratio"] = np.round(df["Renewable_Total_MW"] / np.maximum(df["Demand_MW"], 1.0), 4)
    df["Net_Demand_MW"] = np.round(df["Demand_MW"] - df["Renewable_Total_MW"], 1)
    
    # Clean rounding
    for col in ["Thermal_Coal_MW", "Hydro_MW", "Gas_MW", "Nuclear_MW", "Wind_MW", "Solar_MW", "Demand_MW", "Renewable_Total_MW", "Total_Generation_MW"]:
        df[col] = np.round(df[col], 1)
    df["System_Inertia"] = np.round(df["System_Inertia"], 3)
    
    # Temporal attributes
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["Month"] = df["Timestamp"].dt.month
    
    # Define Target: Grid Operational Stress / Ramp Event (1 = Stress / High Ramp)
    demand_pct_change = df["Demand_MW"].pct_change().fillna(0.0)
    demand_85th = df["Demand_MW"].quantile(0.85)
    
    # Stress occurs during:
    # 1. Steep positive load ramps (> 3.5% in 1 hour)
    # 2. Extreme peak demand regimes (> 85th percentile demand)
    # 3. Sudden net demand spike (demand minus renewable jump > 5%)
    net_demand_pct_change = df["Net_Demand_MW"].pct_change().fillna(0.0)
    
    is_stress = (
        (demand_pct_change > 0.035) | 
        (df["Demand_MW"] >= demand_85th) | 
        (net_demand_pct_change > 0.05)
    ).astype(int)
    
    df["Target_Stress_Event"] = is_stress
    
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "india_electricity_real_hourly.csv")
    df.to_csv(out_path, index=False)
    
    print("\n" + "=" * 80)
    print("SUCCESS: REAL INDIAN NATIONAL GRID (POSOCO / NLDC) DATASET COMPILED!")
    print("=" * 80)
    print(f"Output File:           {out_path}")
    print(f"Total Hourly Records:  {len(df):,} hours ({len(df)/24:.1f} continuous days, {len(df)/(24*7):.1f} weeks)")
    print(f"Date Range:            {df['Timestamp'].min()} -> {df['Timestamp'].max()}")
    print(f"Demand Range:          {df['Demand_MW'].min():,.1f} MW to {df['Demand_MW'].max():,.1f} MW (Mean: {df['Demand_MW'].mean():,.1f} MW)")
    print(f"Thermal Coal Gen:      Mean {df['Thermal_Coal_MW'].mean():,.1f} MW (Max {df['Thermal_Coal_MW'].max():,.1f} MW)")
    print(f"Renewable Peak Gen:    Max Solar {df['Solar_MW'].max():,.1f} MW, Max Wind {df['Wind_MW'].max():,.1f} MW")
    print(f"Renewable Ratio:       Mean {df['Renewable_Ratio'].mean()*100:.2f}% (Peak {df['Renewable_Ratio'].max()*100:.2f}%)")
    print(f"System Inertia:        Mean {df['System_Inertia'].mean():.3f} (Min {df['System_Inertia'].min():.3f}, Max {df['System_Inertia'].max():.3f})")
    print(f"Target Distribution:   {df['Target_Stress_Event'].mean()*100:.2f}% Stress Events ({df['Target_Stress_Event'].sum():,} hours)")
    print(f"File Size:             {os.path.getsize(out_path) / (1024*1024):.2f} MB")
    print("=" * 80)

if __name__ == "__main__":
    main()

