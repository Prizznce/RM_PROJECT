"""
15_validation_comparison.py
============================
Publication-quality comparison: Model Prediction vs Actual 2026 Flood
Clean, noise-filtered figures with professional styling.
"""
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage
import os

year = '2026'
predicted_path = f"data/{year}/flood_risk_map_{year}.tif"
actual_path = f"data/{year}/flood_mask_{year}.tif"
output_dir = "deliverables"
os.makedirs(output_dir, exist_ok=True)

# ---- LOAD DATA ----
print("Loading data...")
with rasterio.open(predicted_path) as src_pred:
    predicted_raw = src_pred.read(1)
    pred_profile = src_pred.profile
    pred_nodata = src_pred.nodata

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

# ---- BUILD CLEAN BINARY MASKS ----
pred_valid = predicted_raw != pred_nodata
predicted_binary = np.zeros_like(predicted_raw, dtype=np.uint8)
predicted_binary[pred_valid & (predicted_raw >= 85.0)] = 1

actual_binary = (actual_resampled >= 0.5).astype(np.uint8)

# ---- SPATIAL DENOISING ----
# Remove small scattered objects (< 200 pixels) from both maps
# This eliminates the speckle noise and makes shapes clean
MIN_OBJ_SIZE = 200

def clean_binary_mask(mask, min_size):
    """Remove small connected components smaller than min_size pixels."""
    labeled, num_features = ndimage.label(mask)
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    too_small = np.array([s < min_size for s in component_sizes])
    remove_mask = too_small[labeled - 1]
    remove_mask[labeled == 0] = False
    cleaned = mask.copy()
    cleaned[remove_mask] = 0
    return cleaned

print("Cleaning predicted mask (removing speckle)...")
predicted_clean = clean_binary_mask(predicted_binary, MIN_OBJ_SIZE)

print("Cleaning actual mask (removing SAR noise)...")
actual_clean = clean_binary_mask(actual_binary, MIN_OBJ_SIZE)

# ---- CROP to the valid data region (remove empty border) ----
rows_with_data = np.any(pred_valid, axis=1)
cols_with_data = np.any(pred_valid, axis=0)
r_min, r_max = np.where(rows_with_data)[0][[0, -1]]
c_min, c_max = np.where(cols_with_data)[0][[0, -1]]
# Add small padding
pad = 20
r_min = max(0, r_min - pad)
r_max = min(predicted_raw.shape[0], r_max + pad)
c_min = max(0, c_min - pad)
c_max = min(predicted_raw.shape[1], c_max + pad)

predicted_crop = predicted_clean[r_min:r_max, c_min:c_max]
actual_crop = actual_clean[r_min:r_max, c_min:c_max]
valid_crop = pred_valid[r_min:r_max, c_min:c_max]

# ---- STYLE SETTINGS ----
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#0d1117',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
})

BG_COLOR = '#0d1117'
FLOOD_COLOR = '#ff4757'
DRY_COLOR = '#1e2a3a'

# ============================================================
# FIGURE 1: Side-by-Side Predicted vs Actual
# ============================================================
print("Generating side-by-side comparison...")
flood_cmap = ListedColormap([DRY_COLOR, FLOOD_COLOR])

fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor=BG_COLOR)
fig.suptitle('FLOOD SUSCEPTIBILITY MODEL VALIDATION  -  Prayagraj 2026',
             fontsize=18, fontweight='bold', color='white', y=1.02)

# Left: Predicted
pred_display = np.ma.masked_where(~valid_crop, predicted_crop)
axes[0].imshow(pred_display, cmap=flood_cmap, vmin=0, vmax=1, interpolation='nearest')
axes[0].set_title('MODEL PREDICTION\nRandom Forest  |  85% Confidence Threshold',
                   fontsize=13, fontweight='bold', color='#58a6ff', pad=12)
axes[0].axis('off')

# Right: Actual
actual_display = np.ma.masked_where(~valid_crop, actual_crop)
axes[1].imshow(actual_display, cmap=flood_cmap, vmin=0, vmax=1, interpolation='nearest')
axes[1].set_title('OBSERVED FLOOD\nSentinel-1 SAR Ground Truth  |  Sep 2026',
                   fontsize=13, fontweight='bold', color='#58a6ff', pad=12)
axes[1].axis('off')

# Shared legend
legend_patches = [
    Patch(facecolor=DRY_COLOR, edgecolor='#555', label='  Dry Land'),
    Patch(facecolor=FLOOD_COLOR, edgecolor='#555', label='  Flood / Inundation'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=2,
           fontsize=13, frameon=True, facecolor='#161b22', edgecolor='#30363d',
           labelcolor='white', handlelength=2, handleheight=1.5,
           bbox_to_anchor=(0.5, 0.02))

plt.subplots_adjust(wspace=0.08, top=0.85, bottom=0.08)
path1 = os.path.join(output_dir, 'comparison_predicted_vs_actual.png')
fig.savefig(path1, dpi=250, bbox_inches='tight', facecolor=BG_COLOR)
print(f"  Saved: {path1}")
plt.close()

# ============================================================
# FIGURE 2: Agreement / Error Map
# ============================================================
print("Generating agreement map...")

# Build categories
# 0 = True Negative, 1 = True Positive, 2 = False Positive, 3 = False Negative
agreement = np.full(predicted_crop.shape, -1, dtype=np.int8)

both_valid = valid_crop
p = predicted_crop[both_valid]
a = actual_crop[both_valid]
indices = np.where(both_valid)

tn_m = (p == 0) & (a == 0)
tp_m = (p == 1) & (a == 1)
fp_m = (p == 1) & (a == 0)
fn_m = (p == 0) & (a == 1)

agreement[indices[0][tn_m], indices[1][tn_m]] = 0
agreement[indices[0][tp_m], indices[1][tp_m]] = 1
agreement[indices[0][fp_m], indices[1][fp_m]] = 2
agreement[indices[0][fn_m], indices[1][fn_m]] = 3

tp_count = tp_m.sum()
fp_count = fp_m.sum()
fn_count = fn_m.sum()
tn_count = tn_m.sum()
total_valid = tp_count + fp_count + fn_count + tn_count

print(f"  True Positive  (Correct flood):    {tp_count:>10,}  ({tp_count/total_valid*100:.1f}%)")
print(f"  True Negative  (Correct dry):      {tn_count:>10,}  ({tn_count/total_valid*100:.1f}%)")
print(f"  False Positive (Over-prediction):  {fp_count:>10,}  ({fp_count/total_valid*100:.1f}%)")
print(f"  False Negative (Missed flood):     {fn_count:>10,}  ({fn_count/total_valid*100:.1f}%)")
spatial_acc = (tp_count + tn_count) / total_valid * 100
print(f"  Spatial Accuracy:                  {spatial_acc:.1f}%")

agree_cmap = ListedColormap(['#161b22', '#2ecc71', '#f39c12', '#3498db'])
agree_display = np.ma.masked_where(agreement == -1, agreement)

fig2, ax2 = plt.subplots(1, 1, figsize=(14, 11), facecolor=BG_COLOR)
ax2.imshow(agree_display, cmap=agree_cmap, vmin=0, vmax=3, interpolation='nearest')
ax2.set_title('PREDICTION vs REALITY  -  Spatial Agreement Map\nPrayagraj Flood 2026',
              fontsize=16, fontweight='bold', color='white', pad=15)
ax2.axis('off')

legend_patches = [
    Patch(facecolor='#161b22', edgecolor='#555',
          label=f'  True Negative  -  Correct Dry  ({tn_count/total_valid*100:.1f}%)'),
    Patch(facecolor='#2ecc71', edgecolor='#555',
          label=f'  True Positive  -  Correct Flood  ({tp_count/total_valid*100:.1f}%)'),
    Patch(facecolor='#f39c12', edgecolor='#555',
          label=f'  False Positive  -  Over-predicted  ({fp_count/total_valid*100:.1f}%)'),
    Patch(facecolor='#3498db', edgecolor='#555',
          label=f'  False Negative  -  Missed Flood  ({fn_count/total_valid*100:.1f}%)'),
]
ax2.legend(handles=legend_patches, loc='lower left', fontsize=11,
           frameon=True, facecolor='#161b22', edgecolor='#30363d',
           labelcolor='white', handlelength=2, handleheight=1.5,
           borderpad=1, labelspacing=0.8)

plt.tight_layout(pad=2)
path2 = os.path.join(output_dir, 'agreement_map_2026.png')
fig2.savefig(path2, dpi=250, bbox_inches='tight', facecolor=BG_COLOR)
print(f"  Saved: {path2}")
plt.close()

print("\nDone! Publication-quality figures saved.")
