"""
Clean and preprocess the Kaggle Hourly Energy Demand & Generation dataset.

Input: energy_dataset.csv/energy_dataset.csv (or energy_dataset.csv.zip)
Output: data/energy_dataset_clean.csv
"""

import os
import pandas as pd
import numpy as np

def clean_energy_dataset(
    raw_path="energy_dataset.csv/energy_dataset.csv",
    output_path="data/energy_dataset_clean.csv"
):
    print(f"Reading raw dataset from: {raw_path}")
    df = pd.read_csv(raw_path)

    # 1. Standardize Timestamps
    # Time string format: '2015-01-01 00:00:00+01:00'
    # Parse UTC or local datetime without timezone offset string for clean timestamp indexing
    parsed_dt = pd.to_datetime(df['time'], utc=True).dt.tz_convert('CET')
    df['Timestamp'] = parsed_dt.dt.strftime('%Y-%m-%d %H:%M:%S')

    # 2. Drop 100% NaN columns
    empty_cols = [c for c in df.columns if df[c].isnull().all()]
    print(f"Dropping 100% empty columns: {empty_cols}")
    df = df.drop(columns=empty_cols)

    # 3. Drop all-zero inactive generation columns
    zero_cols = [
        'generation fossil coal-derived gas',
        'generation fossil oil shale',
        'generation fossil peat',
        'generation geothermal',
        'generation marine',
        'generation wind offshore'
    ]
    existing_zero_cols = [c for c in zero_cols if c in df.columns]
    print(f"Dropping all-zero unused technology columns: {existing_zero_cols}")
    df = df.drop(columns=existing_zero_cols)

    # 4. Handle Missing Values via Linear Time Interpolation + backward/forward fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nulls_before = df[numeric_cols].isnull().sum().sum()
    print(f"Total missing values before interpolation: {nulls_before}")

    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').bfill().ffill()
    nulls_after = df[numeric_cols].isnull().sum().sum()
    print(f"Total missing values after interpolation: {nulls_after}")

    # 5. Define active generation sources contributing to Total Generation
    gen_sources = [
        'generation nuclear',
        'generation fossil gas',
        'generation fossil hard coal',
        'generation fossil brown coal/lignite',
        'generation fossil oil',
        'generation hydro water reservoir',
        'generation hydro run-of-river and poundage',
        'generation solar',
        'generation wind onshore',
        'generation biomass',
        'generation waste',
        'generation other',
        'generation other renewable'
    ]
    df['Total Generation (MW)'] = df[gen_sources].sum(axis=1)

    # 6. Rename columns to standardized, human-readable names with units
    # Keeping Demand (MW), Wind (MW), Solar (MW), Total Generation (MW) backward compatible
    column_mapping = {
        'total load actual': 'Demand (MW)',
        'total load forecast': 'Demand Forecast (MW)',
        'generation wind onshore': 'Wind (MW)',
        'generation solar': 'Solar (MW)',
        'generation nuclear': 'Nuclear (MW)',
        'generation fossil gas': 'Gas (MW)',
        'generation fossil hard coal': 'Hard Coal (MW)',
        'generation fossil brown coal/lignite': 'Lignite (MW)',
        'generation fossil oil': 'Oil (MW)',
        'generation hydro water reservoir': 'Hydro Reservoir (MW)',
        'generation hydro run-of-river and poundage': 'Hydro Run-of-River (MW)',
        'generation hydro pumped storage consumption': 'Hydro Pumped Storage Consumption (MW)',
        'generation biomass': 'Biomass (MW)',
        'generation waste': 'Waste (MW)',
        'generation other': 'Other Conventional (MW)',
        'generation other renewable': 'Other Renewable (MW)',
        'price actual': 'Price Actual (EUR/MWh)',
        'price day ahead': 'Price Day Ahead (EUR/MWh)',
        'forecast solar day ahead': 'Solar Forecast Day Ahead (MW)',
        'forecast wind onshore day ahead': 'Wind Onshore Forecast Day Ahead (MW)'
    }

    df = df.rename(columns=column_mapping)

    # 7. Select and order columns logically
    ordered_columns = [
        'Timestamp',
        'Demand (MW)',
        'Demand Forecast (MW)',
        'Total Generation (MW)',
        'Nuclear (MW)',
        'Gas (MW)',
        'Hard Coal (MW)',
        'Lignite (MW)',
        'Oil (MW)',
        'Hydro Reservoir (MW)',
        'Hydro Run-of-River (MW)',
        'Solar (MW)',
        'Wind (MW)',
        'Biomass (MW)',
        'Waste (MW)',
        'Other Conventional (MW)',
        'Other Renewable (MW)',
        'Hydro Pumped Storage Consumption (MW)',
        'Price Actual (EUR/MWh)',
        'Price Day Ahead (EUR/MWh)',
        'Solar Forecast Day Ahead (MW)',
        'Wind Onshore Forecast Day Ahead (MW)'
    ]

    clean_df = df[ordered_columns].copy()

    # 8. Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    print(f"Successfully cleaned dataset saved to: {output_path}")
    print(f"Cleaned shape: {clean_df.shape}")
    print("Columns in cleaned dataset:")
    for col in clean_df.columns:
        print(f" - {col}")

if __name__ == '__main__':
    clean_energy_dataset()

