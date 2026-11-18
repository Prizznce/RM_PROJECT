import math
import pandas as pd
import numpy as np

class ElectricityStreamLoader:
    """
    Online streaming data loader with optimized causal feature engineering.
    Strictly past signals from t-1, t-2, t-24. Guarantees zero future leakage.
    Supports multi-fuel mix, market pricing, and domain grid harmonics.
    """
    def __init__(self, csv_path="data/energy_dataset_clean.csv"):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.df['Parsed_Timestamp'] = pd.to_datetime(self.df['Timestamp'], format='mixed')
        # Ensure strict chronological ordering
        self.df = self.df.sort_values('Parsed_Timestamp').reset_index(drop=True)

    def __len__(self):
        # We lose the first 24 rows for the 24-hr lag feature warmup
        return len(self.df) - 24

    def iter_stream(self, warmup_rows=24):
        """
        Yields (x_t, y_t, timestamp) for t in [warmup_rows, len(df)-1].
        x_t only contains lagged signals up to t-1 and cyclical calendar coordinates.
        y_t is binary direction: 1 if Demand_t > Demand_{t-1} else 0.
        """
        timestamps = self.df['Parsed_Timestamp'].values
        demands = self.df['Demand (MW)'].values
        winds = self.df['Wind (MW)'].values
        solars = self.df['Solar (MW)'].values
        gens = self.df['Total Generation (MW)'].values

        has_nuclear = 'Nuclear (MW)' in self.df.columns
        has_gas = 'Gas (MW)' in self.df.columns
        has_coal = 'Hard Coal (MW)' in self.df.columns
        has_hydro = 'Hydro Reservoir (MW)' in self.df.columns
        has_price = 'Price Actual (EUR/MWh)' in self.df.columns

        nuclears = self.df['Nuclear (MW)'].values if has_nuclear else None
        gases = self.df['Gas (MW)'].values if has_gas else None
        coals = self.df['Hard Coal (MW)'].values if has_coal else None
        hydros = self.df['Hydro Reservoir (MW)'].values if has_hydro else None
        prices = self.df['Price Actual (EUR/MWh)'].values if has_price else None

        n = len(self.df)
        for t in range(warmup_rows, n):
            ts = pd.Timestamp(timestamps[t])
            hour = ts.hour
            dow = ts.day_of_week

            # Strictly past signals from t-1, t-2, t-24
            d_1 = float(demands[t - 1])
            d_2 = float(demands[t - 2])
            d_24 = float(demands[t - 24])
            s_1 = float(solars[t - 1])
            w_1 = float(winds[t - 1])
            g_1 = float(gens[t - 1])

            x_t = {
                'demand_lag_1': d_1,
                'demand_lag_2': d_2,
                'demand_lag_24': d_24,
                'demand_diff_1_2': d_1 - d_2,
                'demand_diff_1_24': d_1 - d_24,  # Diurnal 24-hr momentum
                'solar_lag_1': s_1,
                'wind_lag_1': w_1,
                'gen_lag_1': g_1,
                'net_load_lag_1': d_1 - (s_1 + w_1),
                'renewable_ratio_lag_1': (s_1 + w_1) / (g_1 + 1e-5),  # Renewable penetration
                'hour_sin': float(math.sin(2 * math.pi * hour / 24.0)),
                'hour_cos': float(math.cos(2 * math.pi * hour / 24.0)),
                'day_of_week': int(dow)
            }

            # Optional multi-fuel & price features when available in cleaned dataset
            if has_nuclear:
                x_t['nuclear_lag_1'] = float(nuclears[t - 1])
            if has_gas:
                x_t['gas_lag_1'] = float(gases[t - 1])
            if has_coal:
                x_t['coal_lag_1'] = float(coals[t - 1])
            if has_hydro:
                x_t['hydro_lag_1'] = float(hydros[t - 1])
            if has_price:
                p_1 = float(prices[t - 1])
                p_2 = float(prices[t - 2])
                x_t['price_lag_1'] = p_1
                x_t['price_diff_1_2'] = p_1 - p_2

            # Target at time t: is demand higher than previous hour?
            y_t = int(demands[t] > demands[t - 1])

            yield x_t, y_t, ts
