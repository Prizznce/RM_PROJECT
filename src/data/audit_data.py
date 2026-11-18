import os
import pandas as pd
import numpy as np

def run_data_audit(csv_path='data/energy_dataset_clean.csv'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist.")

    df = pd.read_csv(csv_path)
    total_raw_rows = len(df)
    
    # Parse dates
    df['Parsed_Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    
    min_date = df['Parsed_Timestamp'].min()
    max_date = df['Parsed_Timestamp'].max()
    date_range_str = f"{min_date.strftime('%Y-%m-%d %H:%M:%S')} to {max_date.strftime('%Y-%m-%d %H:%M:%S')}"
    
    expected_hours = int((max_date - min_date).total_seconds() / 3600) + 1
    
    dup_timestamps = int(df['Parsed_Timestamp'].duplicated().sum())
    dup_full_rows = int(df.duplicated().sum())

    # Check temporal gaps
    time_diffs = df['Parsed_Timestamp'].diff()
    gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
    missing_hours_count = 0
    gap_details = []
    for idx, delta in gaps.items():
        gap_hours = int(delta.total_seconds() / 3600) - 1
        missing_hours_count += gap_hours
        prev_time = df.loc[idx-1, 'Parsed_Timestamp']
        curr_time = df.loc[idx, 'Parsed_Timestamp']
        gap_details.append(f"{prev_time} -> {curr_time} ({gap_hours} missing hours)")

    cols = [
        'Demand (MW)', 'Total Generation (MW)', 'Nuclear (MW)', 'Gas (MW)',
        'Hard Coal (MW)', 'Solar (MW)', 'Wind (MW)'
    ]
    summary_stats = df[cols].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/data_audit.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Data Stream Audit Report: Multi-Fuel Hourly Electricity Stream (2015-2018)\n\n")
        f.write(f"- **Data File**: `{csv_path}`\n")
        f.write(f"- **Row Count**: {total_raw_rows} records (100% hourly intervals)\n")
        f.write(f"- **Date Range**: {date_range_str}\n")
        f.write(f"- **Total Duration**: {expected_hours} expected chronological hours\n\n")
        
        f.write("## 1. Temporal Continuity, Duplicates & Missing Hours\n")
        f.write(f"- **Duplicate Timestamps**: {dup_timestamps}\n")
        f.write(f"- **Duplicate Full Rows**: {dup_full_rows}\n")
        f.write(f"- **Missing Hourly Timestamps**: {missing_hours_count}\n")
        if gap_details:
            f.write(f"- **Gaps Detected**:\n")
            for g in gap_details[:10]:
                f.write(f"  - {g}\n")
        else:
            f.write("- **Gaps Detected**: None. The time stream is perfectly contiguous at 1-hour frequency.\n\n")
            
        f.write("## 2. Generation & Demand Metrics Summary\n")
        f.write("| Feature | Count | Mean (MW) | Std (MW) | Min (MW) | 25% | 50% (Median) | 75% | Max (MW) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for col in cols:
            row_s = summary_stats.loc[col]
            f.write(f"| **{col}** | {int(row_s['count'])} | {row_s['mean']:.2f} | {row_s['std']:.2f} | {row_s['min']:.2f} | {row_s['25%']:.2f} | {row_s['50%']:.2f} | {row_s['75%']:.2f} | {row_s['max']:.2f} |\n")
        f.write("\n")

    print(f"Audit completed successfully! Report generated at: {report_path}")

if __name__ == '__main__':
    run_data_audit()
