"""
18_final_prediction_showcase.py
================================
Creates a polished 3-panel figure showing:
  Panel 1: Pre-flood conditions (NDWI - what we feed the model)
  Panel 2: Model's PREDICTION of flood zones
  Panel 3: ACTUAL flood (SAR ground truth)
This demonstrates the full prediction pipeline for the research paper.
"""
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import ndimage
import os

year = '2026'
output_dir = "deliverables"
os.makedirs(output_dir, exist_ok=True)

# ---- LOAD ALL DATA ----
print("Loading data for 3-panel showcase...")

# Pre-flood optical (for NDWI display)
optical_path = f"data/{year}/prayagraj_preflood_S2_{year}.tif"
predicted_path = f"data/{year}/flood_risk_map_{year}.tif"
actual_path = f"data/{year}/flood_mask_{year}.tif"

with rasterio.open(predicted_path) as src_pred:
    predicted_raw = src_pred.read(1)
    pred_profile = src_pred.profile
    pred_nodata = src_pred.nodata

# Compute NDWI from optical
with rasterio.open(optical_path) as src_opt:
    green_2d = src_opt.read(2).astype(np.float64)
    nir_2d = src_opt.read(4).astype(np.float64)
    ndwi_2d = (green_2d - nir_2d) / (green_2d + nir_2d + 1e-8)
    # Resample to match prediction grid
    ndwi_resampled = np.empty(predicted_raw.shape, dtype=np.float32)
    reproject(
        source=ndwi_2d.astype(np.float32),
        destination=ndwi_resampled,
        src_transform=src_opt.transform,
        src_crs=src_opt.crs,
        dst_transform=pred_profile['transform'],
        dst_crs=pred_profile['crs'],
        resampling=Resampling.nearest
    )

with rasterio.open(actual_path) as src_actual:
    actual_resampled = np.empty(predicted_raw.shape, dtype=np.float32)
    reproject(
        source=rasterio.band(src_actual, 1),
        destination=actual_resampled,
        src_transform=src_actual.transform,
        src_crs=src_actual.crs,
        dst_transform=pred_profile['transform'],
        dst_crs=pred_profile['crs'],
        resampling=Resampling.nearest
    )

# ---- CLEAN DATA ----
pred_valid = predicted_raw != pred_nodata
predicted_binary = np.zeros_like(predicted_raw, dtype=np.uint8)
predicted_binary[pred_valid & (predicted_raw >= 85.0)] = 1
actual_binary = (actual_resampled >= 0.5).astype(np.uint8)

MIN_OBJ_SIZE = 200

def clean_mask(mask, min_size):
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    too_small = np.array([s < min_size for s in sizes])
    remove = too_small[labeled - 1]
    remove[labeled == 0] = False
    cleaned = mask.copy()
    cleaned[remove] = 0
    return cleaned

predicted_clean = clean_mask(predicted_binary, MIN_OBJ_SIZE)
actual_clean = clean_mask(actual_binary, MIN_OBJ_SIZE)

# ---- CROP ----
rows = np.any(pred_valid, axis=1)
cols = np.any(pred_valid, axis=0)
r_min, r_max = np.where(rows)[0][[0, -1]]
c_min, c_max = np.where(cols)[0][[0, -1]]
pad = 20
r_min, r_max = max(0, r_min-pad), min(predicted_raw.shape[0], r_max+pad)
c_min, c_max = max(0, c_min-pad), min(predicted_raw.shape[1], c_max+pad)

ndwi_crop = ndwi_resampled[r_min:r_max, c_min:c_max]
pred_crop = predicted_clean[r_min:r_max, c_min:c_max]
actual_crop = actual_clean[r_min:r_max, c_min:c_max]
valid_crop = pred_valid[r_min:r_max, c_min:c_max]

# ---- STYLE ----
BG_COLOR = '#0d1117'
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'text.color': 'white',
})

# ---- 3-PANEL FIGURE ----
print("Generating 3-panel prediction showcase...")
fig, axes = plt.subplots(1, 3, figsize=(24, 9), facecolor=BG_COLOR)
fig.suptitle('FLOOD PREDICTION PIPELINE  -  Prayagraj 2026',
             fontsize=20, fontweight='bold', color='white', y=1.03)

# Panel 1: Pre-Flood NDWI (INPUT)
ndwi_display = np.ma.masked_where(~valid_crop, ndwi_crop)
ndwi_cmap = LinearSegmentedColormap.from_list('ndwi', 
    ['#8B4513', '#D2691E', '#F5DEB3', '#90EE90', '#228B22', '#006400', '#004080', '#0000FF'])
im1 = axes[0].imshow(ndwi_display, cmap=ndwi_cmap, vmin=-0.3, vmax=0.5, interpolation='nearest')
axes[0].set_title('STEP 1: INPUT\nPre-Monsoon NDWI (Jun 2026)',
                   fontsize=13, fontweight='bold', color='#ffa657', pad=15)
axes[0].axis('off')
# NDWI colorbar
cbar = fig.colorbar(im1, ax=axes[0], fraction=0.03, pad=0.02, aspect=30)
cbar.set_label('NDWI', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=9)

# Panel 2: Model Prediction (OUTPUT)
flood_cmap = ListedColormap(['#1e2a3a', '#ff4757'])
pred_display = np.ma.masked_where(~valid_crop, pred_crop)
axes[1].imshow(pred_display, cmap=flood_cmap, vmin=0, vmax=1, interpolation='nearest')
axes[1].set_title('STEP 2: PREDICTION\nRandom Forest Model (85% Threshold)',
                   fontsize=13, fontweight='bold', color='#ff4757', pad=15)
axes[1].axis('off')
pred_legend = [
    Patch(facecolor='#1e2a3a', edgecolor='#555', label='  Safe'),
    Patch(facecolor='#ff4757', edgecolor='#555', label='  Flood Predicted'),
]
axes[1].legend(handles=pred_legend, loc='lower left', fontsize=9,
               frameon=True, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

# Panel 3: Actual Flood (VALIDATION)
actual_display = np.ma.masked_where(~valid_crop, actual_crop)
axes[2].imshow(actual_display, cmap=flood_cmap, vmin=0, vmax=1, interpolation='nearest')
axes[2].set_title('STEP 3: VALIDATION\nActual SAR-Observed Flood (Sep 2026)',
                   fontsize=13, fontweight='bold', color='#2ecc71', pad=15)
axes[2].axis('off')
actual_legend = [
    Patch(facecolor='#1e2a3a', edgecolor='#555', label='  Dry'),
    Patch(facecolor='#ff4757', edgecolor='#555', label='  Actual Flood'),
]
axes[2].legend(handles=actual_legend, loc='lower left', fontsize=9,
               frameon=True, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

# Stats box
stats_text = ("Model Performance\n"
              "Accuracy: 79.7%\n"
              "ROC AUC: 0.9811\n"
              "False Positive: 1.0%")
fig.text(0.5, 0.02, stats_text, ha='center', va='bottom',
         fontsize=11, color='white', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#161b22', edgecolor='#58a6ff', alpha=0.9))

plt.subplots_adjust(wspace=0.08, top=0.82, bottom=0.10)
path = os.path.join(output_dir, 'flood_prediction_pipeline.png')
fig.savefig(path, dpi=250, bbox_inches='tight', facecolor=BG_COLOR)
print(f"  Saved: {path}")
plt.close()

print("Done!")
