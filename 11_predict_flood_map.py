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
master_ref_path = f"data/{year}/prayagraj_preflood_S2_{year}.tif"
static_topo = "data/prayagraj_topo_features.tif"
static_dist = "data/prayagraj_distance_to_river.tif"
static_twi  = "data/prayagraj_twi.tif"
static_lulc = "data/prayagraj_LULC_ESA_WorldCover.tif"
static_soil = "data/prayagraj_soil_clay_pct.tif"

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
    twi = load_and_resample(static_twi, master, band=1)
    lulc = load_and_resample(static_lulc, master, band=1)
    soil_clay = load_and_resample(static_soil, master, band=1)
    
    # Calculate NDWI
    green = master.read(2).astype(float).flatten()
    nir = master.read(4).astype(float).flatten()
    ndwi = (green - nir) / (green + nir + 1e-8)

    # Build a valid-data mask
    band1 = master.read(1).astype(float).flatten()
    has_satellite = (green != 0) | (nir != 0) | (band1 != 0)
    has_topo = (elevation != 0)
    valid_mask = has_satellite & has_topo
    print(f"Valid pixels: {valid_mask.sum():,} / {len(valid_mask):,} "
          f"({valid_mask.sum()/len(valid_mask)*100:.1f}%)")

print("Constructing feature matrix...")
df = pd.DataFrame({
    'Elevation': elevation,
    'Slope': slope,
    'Distance_To_River': distance,
    'TWI': twi,
    'LULC': lulc,
    'Soil_Clay_Pct': soil_clay,
    'Pre_Flood_NDWI': ndwi
})

# Handle NaNs to prevent prediction errors
df.fillna(0, inplace=True)
df.replace([np.inf, -np.inf], 0, inplace=True)

print("Predicting flood probability (0-100%)...")
# predict_proba returns [prob_class_0, prob_class_1]
probabilities = rf_model.predict_proba(df)[:, 1]

# Convert to percentage
prob_map = (probabilities * 100).astype(np.float32)

# Mask out nodata areas (outside satellite/topo footprint)
NODATA_VAL = -9999.0
prob_map[~valid_mask] = NODATA_VAL

valid_pixels = prob_map[prob_map != NODATA_VAL]
print(f"Probability distribution:")
print(f"  0-20% (Very Low):   {((valid_pixels >= 0) & (valid_pixels < 20)).sum():>8,}")
print(f"  20-40% (Low):       {((valid_pixels >= 20) & (valid_pixels < 40)).sum():>8,}")
print(f"  40-60% (Moderate):  {((valid_pixels >= 40) & (valid_pixels < 60)).sum():>8,}")
print(f"  60-80% (High):      {((valid_pixels >= 60) & (valid_pixels < 80)).sum():>8,}")
print(f"  80-100% (Very High):{((valid_pixels >= 80)).sum():>8,}")

prob_map_2d = prob_map.reshape(shape)

# Save FULL probability gradient (for susceptibility heatmap)
output_path = f"data/{year}/flood_susceptibility_map_{year}.tif"
profile.update(
    dtype=rasterio.float32,
    count=1,
    compress='lzw',
    nodata=NODATA_VAL
)

print(f"Saving susceptibility map to {output_path}...")
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(prob_map_2d, 1)

# Also save thresholded version (for binary comparison maps)
PROB_THRESHOLD = 85.0
prob_map_thresh = prob_map.copy()
prob_map_thresh[(prob_map_thresh < PROB_THRESHOLD) & (prob_map_thresh != NODATA_VAL)] = NODATA_VAL
prob_map_thresh_2d = prob_map_thresh.reshape(shape)

output_path_thresh = f"data/{year}/flood_risk_map_{year}.tif"
print(f"Saving thresholded risk map to {output_path_thresh}...")
with rasterio.open(output_path_thresh, 'w', **profile) as dst:
    dst.write(prob_map_thresh_2d, 1)

print(f"High-risk flood pixels (>{PROB_THRESHOLD}%): "
      f"{(prob_map_thresh != NODATA_VAL).sum():,}")
print("Done! Both maps generated.")

