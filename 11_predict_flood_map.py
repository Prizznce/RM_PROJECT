import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

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

year = '2026'
print(f"Generating Spatial Probability Heat Map for {year}...")

# Use the pre-flood optical as the master spatial reference for 2026
master_ref_path = f"data/prayagraj_preflood_S2_{year}.tif"
static_topo = "data/prayagraj_topo_features.tif"
static_dist = "data/prayagraj_distance_to_river.tif"
precip_path = f"data/prayagraj_precip_10d_{year}.tif"

model_path = 'predictive_flood_rf_model.joblib'

if not os.path.exists(model_path):
    print("Model not found! Run 09_train_ml_model.py first.")
    exit()

rf_model = joblib.load(model_path)

with rasterio.open(master_ref_path) as master:
    profile = master.profile
    shape = master.shape
    
    print("Loading and aligning feature layers (this may take a moment)...")
    elevation = load_and_resample(static_topo, master, band=1)
    slope = load_and_resample(static_topo, master, band=2)
    distance = load_and_resample(static_dist, master, band=1)
    rainfall = load_and_resample(precip_path, master, band=1)
    
    # Calculate NDWI
    green = master.read(2).astype(float).flatten()
    nir = master.read(4).astype(float).flatten()
    ndwi = (green - nir) / (green + nir + 1e-8)

print("Constructing feature matrix...")
df = pd.DataFrame({
    'Elevation': elevation,
    'Slope': slope,
    'Distance_To_River': distance,
    'Pre_Flood_NDWI': ndwi
})

# Handle NaNs to prevent prediction errors
df.fillna(0, inplace=True)
df.replace([np.inf, -np.inf], 0, inplace=True)

print("Predicting flood probability (0-100%)...")
# predict_proba returns [prob_class_0, prob_class_1]
probabilities = rf_model.predict_proba(df)[:, 1]

# Convert to percentage and reshape back to 2D
prob_map_2d = (probabilities * 100).reshape(shape).astype(np.float32)

# Save to GeoTIFF
output_path = f"data/flood_risk_map_{year}.tif"
profile.update(
    dtype=rasterio.float32,
    count=1,
    compress='lzw'
)

print(f"Saving risk map to {output_path}...")
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(prob_map_2d, 1)

print("Heat map generated successfully! Ready for QGIS visualization.")
