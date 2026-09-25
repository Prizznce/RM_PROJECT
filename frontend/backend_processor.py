import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def run_live_prediction(uploaded_tif_path, output_png_path, progress_callback=None):
    if progress_callback: progress_callback(10, "Loading static terrain features...")
    
    static_topo = os.path.join(BASE_DIR, "data", "prayagraj_topo_features.tif")
    static_dist = os.path.join(BASE_DIR, "data", "prayagraj_distance_to_river.tif")
    static_twi  = os.path.join(BASE_DIR, "data", "prayagraj_twi.tif")
    static_lulc = os.path.join(BASE_DIR, "data", "prayagraj_LULC_ESA_WorldCover.tif")
    static_soil = os.path.join(BASE_DIR, "data", "prayagraj_soil_clay_pct.tif")
    model_path = os.path.join(BASE_DIR, 'predictive_flood_rf_model.joblib')
    
    if not os.path.exists(model_path):
        # Fallback to the best model if rf_model is missing
        model_path = os.path.join(BASE_DIR, 'predictive_flood_best_model.joblib')

    rf_model = joblib.load(model_path)
    
    with rasterio.open(uploaded_tif_path) as master:
        profile = master.profile
        shape = master.shape
        bounds = master.bounds
        
        elevation = load_and_resample(static_topo, master, band=1)
        slope = load_and_resample(static_topo, master, band=2)
        distance = load_and_resample(static_dist, master, band=1)
        twi = load_and_resample(static_twi, master, band=1)
        lulc = load_and_resample(static_lulc, master, band=1)
        soil_clay = load_and_resample(static_soil, master, band=1)
        
        if progress_callback: progress_callback(40, "Calculating NDWI from uploaded imagery...")
        # Assuming Sentinel-2 bands: B3 (Green) is band 2, B8 (NIR) is band 4
        # Adjust if the uploaded tif has different band mappings
        green = master.read(2).astype(float).flatten()
        nir = master.read(4).astype(float).flatten()
        ndwi = (green - nir) / (green + nir + 1e-8)
        
        band1 = master.read(1).astype(float).flatten()
        has_satellite = (green != 0) | (nir != 0) | (band1 != 0)
        has_topo = (elevation != 0)
        valid_mask = has_satellite & has_topo
        
    if progress_callback: progress_callback(60, "Running Random Forest model predictions...")
    
    df = pd.DataFrame({
        'Elevation': elevation,
        'Slope': slope,
        'Distance_To_River': distance,
        'TWI': twi,
        'LULC': lulc,
        'Soil_Clay_Pct': soil_clay,
        'Pre_Flood_NDWI': ndwi
    })
    
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    probabilities = rf_model.predict_proba(df)[:, 1]
    prob_map = (probabilities * 100).astype(np.float32)
    
    NODATA_VAL = -9999.0
    prob_map[~valid_mask] = NODATA_VAL
    prob_map_2d = prob_map.reshape(shape)
    
    if progress_callback: progress_callback(85, "Rendering continuous susceptibility heatmap...")
    
    # Generate Map Image
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    
    prob_plot = np.where(prob_map_2d == NODATA_VAL, np.nan, prob_map_2d)
    
    # Custom colormap: Transparent/Blueish to Yellow to Bright Red
    colors = [(0, 0, 0, 0), (0.2, 0.4, 0.8, 0.3), (0.9, 0.8, 0.2, 0.7), (1, 0.1, 0.1, 1)]
    cmap = mcolors.LinearSegmentedColormap.from_list("RiskGradient", colors)
    
    im = ax.imshow(prob_plot, cmap=cmap, extent=extent, vmin=0, vmax=100)
    
    ax.set_title("Live Predicted Flood Susceptibility", fontsize=16, color='white', pad=20, fontweight='bold')
    ax.tick_params(colors='white')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flood Probability (%)', color='white', size=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    plt.savefig(output_png_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    
    if progress_callback: progress_callback(100, "Done!")
    return True
