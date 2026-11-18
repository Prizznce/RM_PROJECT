"""
Industrial Cost, Energy & Carbon ROI Calculator.
Translates stream mining CPU time savings into concrete economic and sustainability metrics.
"""

import os
import pandas as pd

def compute_roi_and_sustainability(
    summary_path="results/tables/benchmark_summary.csv",
    output_path="results/tables/roi_and_sustainability.csv",
    edge_nodes=1000,
    readings_per_day=24,
    edge_power_watts=65.0,  # Typical industrial IoT / edge node TDP
    kwh_cost_usd=0.15,      # Industrial electricity rate ($/kWh)
    grid_carbon_intensity=0.72,  # kg CO2 / kWh (National grid average)
    cloud_inference_cost_per_million=1.50  # AWS Lambda / GCP Cloud Function inference cost
):
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"{summary_path} does not exist.")

    df = pd.read_csv(summary_path)
    arf_row = df[df['model_name'] == 'Always-ARF'].iloc[0]
    arf_latency_us = arf_row['mean_cpu_us_per_instance']

    annual_inferences_per_node = readings_per_day * 365
    total_fleet_annual_inferences = edge_nodes * annual_inferences_per_node

    results = []
    for _, row in df.iterrows():
        m_name = row['model_name']
        lat_us = row['mean_cpu_us_per_instance']
        acc = row['accuracy_pct']
        
        # CPU compute time per year across entire edge fleet
        total_cpu_seconds = (total_fleet_annual_inferences * lat_us) * 1e-6
        total_cpu_hours = total_cpu_seconds / 3600.0
        
        # Energy consumed by processor for inference (kWh)
        kwh_consumed = (total_cpu_seconds / 3600.0) * (edge_power_watts / 1000.0)
        electricity_cost_usd = kwh_consumed * kwh_cost_usd
        co2_emissions_kg = kwh_consumed * grid_carbon_intensity

        # Savings relative to Always-ARF
        arf_cpu_hours = ((total_fleet_annual_inferences * arf_latency_us) * 1e-6) / 3600.0
        cpu_hours_saved = arf_cpu_hours - total_cpu_hours
        cpu_saving_pct = row['cpu_saving_pct']
        
        # Carbon saved
        kwh_saved = cpu_hours_saved * (edge_power_watts / 1000.0)
        co2_saved_kg = kwh_saved * grid_carbon_intensity
        cloud_cost_saved_usd = (total_fleet_annual_inferences / 1e6) * cloud_inference_cost_per_million * (cpu_saving_pct / 100.0)

        results.append({
            'Model_Policy': m_name,
            'Accuracy_pct': acc,
            'Latency_us': lat_us,
            'CPU_Saving_pct': cpu_saving_pct,
            'Annual_Fleet_Inferences': total_fleet_annual_inferences,
            'Annual_CPU_Hours': round(total_cpu_hours, 1),
            'Annual_CPU_Hours_Saved': round(cpu_hours_saved, 1),
            'Electricity_Saved_kWh': round(kwh_saved, 1),
            'Power_Cost_Saved_USD': round(kwh_saved * kwh_cost_usd, 2),
            'Cloud_Cost_Saved_USD': round(cloud_cost_saved_usd, 2),
            'CO2_Emissions_Avoided_kg': round(co2_saved_kg, 1)
        })

    roi_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    roi_df.to_csv(output_path, index=False)
    print(f"ROI and Sustainability metrics saved to: {output_path}")
    return roi_df

if __name__ == '__main__':
    compute_roi_and_sustainability()

