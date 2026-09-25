import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd
import os
import joblib
import warnings
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

def load_and_resample(filepath, match_dataset, band=1):
    """
    Loads a raster band and resamples it to match the shape, transform, and crs 
    of the `match_dataset` using Nearest Neighbor interpolation.
    """
    with rasterio.open(filepath) as src:
        resampled_data = np.empty(match_dataset.shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, band),
            destination=resampled_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=match_dataset.transform,
            dst_crs=match_dataset.crs,
            resampling=Resampling.nearest
        )
        return resampled_data.flatten()

print("Loading data for XGBoost weather model...")

years_to_check = ['2019', '2021', '2022', '2024', '2025', '2026']
all_dfs = []

static_topo = "data/prayagraj_topo_features.tif"
static_dist = "data/prayagraj_distance_to_river.tif"

for year in years_to_check:
    label_path = f"data/{year}/flood_mask_{year}.tif"
    precip_7d_path = f"data/{year}/prayagraj_precip_7d_{year}.tif"
    
    if all(map(os.path.exists, [label_path, precip_7d_path, static_topo, static_dist])):
        print(f"Extracting features for year {year}...")
        try:
            with rasterio.open(label_path) as src_label:
                labels = src_label.read(1).flatten()
                elevation = load_and_resample(static_topo, src_label, band=1)
                slope = load_and_resample(static_topo, src_label, band=2)
                distance = load_and_resample(static_dist, src_label, band=1)
                rainfall_7d = load_and_resample(precip_7d_path, src_label, band=1)
                
            df = pd.DataFrame({
                'Elevation': elevation,
                'Slope': slope,
                'Distance_To_River': distance,
                'Rainfall_7d': rainfall_7d,
                'Target': labels,
                'Year': year
            })
            
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # Sub-sample
            water_pixels = df[df['Target'] == 1]
            land_pixels = df[df['Target'] == 0]
            n_samples = min(5000, len(water_pixels), len(land_pixels))
            
            if n_samples > 0:
                water_sampled = water_pixels.sample(n=n_samples, random_state=42)
                land_sampled = land_pixels.sample(n=n_samples, random_state=42)
                event_df = pd.concat([water_sampled, land_sampled])
                all_dfs.append(event_df)
            else:
                print(f"Skipping {year}: not enough valid data.")
        except Exception as e:
            print(f"Error processing year {year}: {e}")
    else:
        print(f"Missing required files for {year}. Skipping.")

combined_df = pd.concat(all_dfs).sample(frac=1, random_state=42)
feature_cols = ['Elevation', 'Slope', 'Distance_To_River', 'Rainfall_7d']

print("\nApplying strict Temporal Hold-Out...")
train_df = combined_df[combined_df['Year'].isin(['2019', '2021'])]
test_df = combined_df[combined_df['Year'] == '2026']

X_train = train_df[feature_cols]
y_train = train_df['Target']
X_test = test_df[feature_cols]
y_test = test_df['Target']

print(f"Training set size (19/21): {len(X_train)}")
print(f"Testing set size (26): {len(X_test)}")

print("\nTraining XGBoost Classifier...")
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

print("\nEvaluating XGBoost Model on 2026 Test Set...")
y_pred = xgb_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Dry (0)', 'Inundated (1)']))

output_path = 'deliverables/xgboost_weather_model.joblib'
joblib.dump(xgb_model, output_path)
print(f"\nModel saved successfully to {output_path}")
