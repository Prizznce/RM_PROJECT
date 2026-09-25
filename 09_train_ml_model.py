import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

def load_and_resample(filepath, match_dataset, band=1):
    """
    Loads a raster band and resamples it to match the shape, transform, and crs 
    of the `match_dataset` using Nearest Neighbor interpolation.
    """
    with rasterio.open(filepath) as src:
        # Initialize an empty array with the target shape
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

print("Building Predictive Feature Matrix across multiple events...")

years_to_check = ['2019', '2021', '2022', '2024', '2025', '2026']
all_dfs = []

static_topo = "data/prayagraj_topo_features.tif"
static_dist = "data/prayagraj_distance_to_river.tif"
static_twi  = "data/prayagraj_twi.tif"
static_lulc = "data/prayagraj_LULC_ESA_WorldCover.tif"
static_soil = "data/prayagraj_soil_clay_pct.tif"

for year in years_to_check:
    label_path = f"data/{year}/flood_mask_{year}.tif"
    precip_path = f"data/{year}/prayagraj_precip_10d_{year}.tif"
    optical_path = f"data/{year}/prayagraj_preflood_S2_{year}.tif"

    if all(map(os.path.exists, [label_path, optical_path, static_topo, static_dist])):
        print(f"\nProcessing event year: {year}")
        try:
            # Load Target Label (this acts as the spatial reference 'match_dataset' for everything else)
            with rasterio.open(label_path) as src_label:
                labels = src_label.read(1).flatten()
                
                # 1. Load Static Features (resampled to match label)
                elevation = load_and_resample(static_topo, src_label, band=1)
                slope = load_and_resample(static_topo, src_label, band=2)
                distance = load_and_resample(static_dist, src_label, band=1)
                twi = load_and_resample(static_twi, src_label, band=1)
                lulc = load_and_resample(static_lulc, src_label, band=1)
                soil_clay = load_and_resample(static_soil, src_label, band=1)
                
                # 2. Load Dynamic Feature: NDWI
                green = load_and_resample(optical_path, src_label, band=2)
                nir = load_and_resample(optical_path, src_label, band=4)
                ndwi = (green - nir) / (green + nir + 1e-8)

            # Build DataFrame
            df = pd.DataFrame({
                'Elevation': elevation,
                'Slope': slope,
                'Distance_To_River': distance,
                'TWI': twi,
                'LULC': lulc,
                'Soil_Clay_Pct': soil_clay,
                'Pre_Flood_NDWI': ndwi,
                'Target': labels,
                'Year': year
            })
            
            # Clean data
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # Sub-sample to keep memory manageable (e.g. 5000 per class per year)
            water_pixels = df[df['Target'] == 1]
            land_pixels = df[df['Target'] == 0]
            n_samples = min(5000, len(water_pixels), len(land_pixels))
            
            if n_samples > 0:
                water_sampled = water_pixels.sample(n=n_samples, random_state=42)
                land_sampled = land_pixels.sample(n=n_samples, random_state=42)
                event_df = pd.concat([water_sampled, land_sampled])
                all_dfs.append(event_df)
                print(f"Added {len(event_df)} balanced pixels from {year} to training set.")
            else:
                print(f"Skipping {year}: not enough valid data to sample.")
                
        except Exception as e:
            print(f"Error processing year {year}: {e}")
    else:
        print(f"Missing required files for {year}. Skipping.")

if not all_dfs:
    print("Error: No valid data found for training.")
    exit()

# Combine all events
print("\nCombining datasets across events...")
combined_df = pd.concat(all_dfs).sample(frac=1, random_state=42)
print(f"Total training pixels: {len(combined_df)}")

# IMPORTANT: SAR is excluded. These are pure predictive precursors.
feature_cols = ['Elevation', 'Slope', 'Distance_To_River', 'TWI', 'LULC', 'Soil_Clay_Pct', 'Pre_Flood_NDWI']
# ----------------------------------------------------
# STRICT TEMPORAL HOLD-OUT VALIDATION
# ----------------------------------------------------
print("\nSplitting data using Temporal Hold-Out...")
# Train on historical events
train_df = combined_df[combined_df['Year'].isin(['2019', '2021', '2022', '2024', '2025'])]
# Test exclusively on the unseen future event (2026)
test_df = combined_df[combined_df['Year'] == '2026']

X_train = train_df[feature_cols]
y_train = train_df['Target']
X_test = test_df[feature_cols]
y_test = test_df['Target']

print(f"Training set size: {len(X_train)} (2019-2025)")
print(f"Testing set size:  {len(X_test)} (2026 only)")
# ----------------------------------------------------

# Train the Predictive Model
print("\nTraining Predictive Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate Model
print("\n--- Model Evaluation ---")
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Dry (0)', 'Inundated (1)']))

importances = rf_model.feature_importances_
print("\nFeature Importances:")
for name, imp in zip(feature_cols, importances):
    print(f"{name.ljust(20)}: {imp:.4f}")

# Save the Model
model_filename = 'predictive_flood_rf_model.joblib'
joblib.dump(rf_model, model_filename)
print(f"\nPredictive model successfully saved as: {model_filename}")
