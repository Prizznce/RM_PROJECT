import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

def load_and_resample(filepath, match_dataset, band=1):
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

print("Loading 2026 test data and Rainfall model for diagnostics...")

# Paths
year = '2026'
label_path = f"data/flood_mask_{year}.tif"
static_topo = "data/prayagraj_topo_features.tif"
static_dist = "data/prayagraj_distance_to_river.tif"
precip_7d_path = f"data/prayagraj_precip_7d_{year}.tif"
precip_30d_path = f"data/prayagraj_precip_30d_{year}.tif"
optical_path = f"data/prayagraj_preflood_S2_{year}.tif"

# Load Data
with rasterio.open(label_path) as src_label:
    labels = src_label.read(1).flatten()
    elevation = load_and_resample(static_topo, src_label, band=1)
    slope = load_and_resample(static_topo, src_label, band=2)
    distance = load_and_resample(static_dist, src_label, band=1)
    rainfall_7d = load_and_resample(precip_7d_path, src_label, band=1)
    rainfall_30d = load_and_resample(precip_30d_path, src_label, band=1)
    green = load_and_resample(optical_path, src_label, band=2)
    nir = load_and_resample(optical_path, src_label, band=4)
    ndwi = (green - nir) / (green + nir + 1e-8)

df = pd.DataFrame({
    'Elevation': elevation,
    'Slope': slope,
    'Distance_To_River': distance,
    'Pre_Flood_NDWI': ndwi,
    'Rainfall_7d': rainfall_7d,
    'Rainfall_30d': rainfall_30d,
    'Target': labels
})

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Sub-sample matching the training script logic
water_pixels = df[df['Target'] == 1]
land_pixels = df[df['Target'] == 0]
n_samples = min(5000, len(water_pixels), len(land_pixels))
water_sampled = water_pixels.sample(n=n_samples, random_state=42)
land_sampled = land_pixels.sample(n=n_samples, random_state=42)
test_df = pd.concat([water_sampled, land_sampled])

feature_cols = ['Elevation', 'Slope', 'Distance_To_River', 'Pre_Flood_NDWI', 'Rainfall_7d', 'Rainfall_30d']
X_test = test_df[feature_cols]
y_test = test_df['Target']

# Load Model and Predict
rf_model = joblib.load('predictive_flood_rf_model_with_rainfall.joblib')
y_pred = rf_model.predict(X_test)

# --- USER DIAGNOSTIC CODE ---

# 1. Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 2. Full precision/recall/F1 breakdown
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 3. Actual class balance in the 2026 test set
unique, counts = np.unique(y_test, return_counts=True)
print("\n2026 test set class balance:")
print(dict(zip(unique, counts)))

# 4. Sanity-check the rainfall feature itself
print("\nRainfall feature stats (2026 test set):")
print("Min:", X_test['Rainfall_7d'].min(), "Max:", X_test['Rainfall_7d'].max())
print("Unique values:", X_test['Rainfall_7d'].nunique())