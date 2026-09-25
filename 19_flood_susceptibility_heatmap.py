"""
19_flood_susceptibility_heatmap.py
===================================
Creates a proper flood SUSCEPTIBILITY map showing the full 0-100%
probability gradient across the entire study area. This shows:
- Green zones: Low flood risk (safe areas)
- Yellow zones: Moderate risk (elevated floodplain) 
- Orange zones: High risk (near-river floodplain)
- Red zones: Very high risk (active flood channel)
"""
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Patch
import os

year = '2026'
output_dir = "deliverables"
os.makedirs(output_dir, exist_ok=True)

BG_COLOR = '#0d1117'

# ---- LOAD SUSCEPTIBILITY MAP ----
susc_path = f"data/{year}/flood_susceptibility_map_{year}.tif"
print(f"Loading susceptibility map: {susc_path}")

with rasterio.open(susc_path) as src:
    susc_data = src.read(1)
    nodata = src.nodata

valid = susc_data != nodata

# ---- AUTO-CROP ----
rows = np.any(valid, axis=1)
cols = np.any(valid, axis=0)
r_min, r_max = np.where(rows)[0][[0, -1]]
c_min, c_max = np.where(cols)[0][[0, -1]]
pad = 15
r_min = max(0, r_min - pad)
r_max = min(susc_data.shape[0], r_max + pad)
c_min = max(0, c_min - pad)
c_max = min(susc_data.shape[1], c_max + pad)

susc_crop = susc_data[r_min:r_max, c_min:c_max]
valid_crop = valid[r_min:r_max, c_min:c_max]

# Mask nodata
susc_display = np.ma.masked_where(~valid_crop, susc_crop)

# ---- CUSTOM COLORMAP: Green -> Yellow -> Orange -> Red ----
colors = [
    '#1a5632',  # 0%  - dark green (very low risk)
    '#2d8a4e',  # 15% - green
    '#5cb85c',  # 30% - light green
    '#f0e130',  # 45% - yellow (moderate risk)
    '#f5a623',  # 60% - orange (high risk)
    '#e74c3c',  # 75% - red
    '#c0392b',  # 90% - dark red (very high risk)
    '#8e0000',  # 100% - deep crimson
]
susc_cmap = LinearSegmentedColormap.from_list('susceptibility', colors, N=256)
susc_cmap.set_bad(color=BG_COLOR)

# ---- STYLE ----
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'text.color': 'white',
})

# ---- FIGURE ----
print("Generating flood susceptibility heatmap...")
fig, ax = plt.subplots(1, 1, figsize=(14, 12), facecolor=BG_COLOR)

fig.suptitle('FLOOD SUSCEPTIBILITY MAP  -  Prayagraj 2026',
             fontsize=20, fontweight='bold', color='white', y=0.97)
ax.set_title('Random Forest Model  |  7 Features  |  Trained on 2019-2025',
             fontsize=12, color='#8b949e', pad=12)

im = ax.imshow(susc_display, cmap=susc_cmap, vmin=0, vmax=100, interpolation='nearest')
ax.axis('off')

# ---- COLORBAR ----
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, aspect=35)
cbar.set_label('Flood Susceptibility (%)', color='white', fontsize=13, fontweight='bold')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=11)

# Add zone labels on colorbar
cbar.ax.text(1.5, 10, 'Very Low', va='center', ha='left', 
             color='#2d8a4e', fontsize=9, fontweight='bold', transform=cbar.ax.transData)
cbar.ax.text(1.5, 30, 'Low', va='center', ha='left',
             color='#5cb85c', fontsize=9, fontweight='bold', transform=cbar.ax.transData)
cbar.ax.text(1.5, 50, 'Moderate', va='center', ha='left',
             color='#f0e130', fontsize=9, fontweight='bold', transform=cbar.ax.transData)
cbar.ax.text(1.5, 70, 'High', va='center', ha='left',
             color='#f5a623', fontsize=9, fontweight='bold', transform=cbar.ax.transData)
cbar.ax.text(1.5, 90, 'Very High', va='center', ha='left',
             color='#c0392b', fontsize=9, fontweight='bold', transform=cbar.ax.transData)

# ---- STATS BOX ----
valid_probs = susc_crop[valid_crop]
stats_lines = [
    f"Total area analyzed: {valid_crop.sum():,} pixels",
    f"Very Low  (0-20%):  {((valid_probs < 20)).sum():>8,}  ({(valid_probs < 20).mean()*100:.1f}%)",
    f"Low       (20-40%): {((valid_probs >= 20) & (valid_probs < 40)).sum():>8,}  ({((valid_probs >= 20) & (valid_probs < 40)).mean()*100:.1f}%)",
    f"Moderate  (40-60%): {((valid_probs >= 40) & (valid_probs < 60)).sum():>8,}  ({((valid_probs >= 40) & (valid_probs < 60)).mean()*100:.1f}%)",
    f"High      (60-80%): {((valid_probs >= 60) & (valid_probs < 80)).sum():>8,}  ({((valid_probs >= 60) & (valid_probs < 80)).mean()*100:.1f}%)",
    f"Very High (80-100%):{((valid_probs >= 80)).sum():>8,}  ({(valid_probs >= 80).mean()*100:.1f}%)",
]
stats_text = '\n'.join(stats_lines)
fig.text(0.03, 0.02, stats_text, ha='left', va='bottom',
         fontsize=9, color='#c9d1d9', family='monospace',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#161b22', 
                   edgecolor='#30363d', alpha=0.95))

plt.subplots_adjust(top=0.90, bottom=0.08)
path = os.path.join(output_dir, 'flood_susceptibility_heatmap.png')
fig.savefig(path, dpi=250, bbox_inches='tight', facecolor=BG_COLOR)
print(f"  Saved: {path}")
plt.close()

print("Done!")
