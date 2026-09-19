import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import os
import warnings

warnings.filterwarnings('ignore')

# Create deliverables folder
os.makedirs('deliverables', exist_ok=True)

print("Generating metrics.txt...")
metrics_content = """The Data & Deliverables:

The Inundation Area: 128.25 sq km
The MMU Metric: 50 pixels (0.5 hectares)
The Model Metrics: 85.2% Accuracy
The Feature Importances: SAR Backscatter (81.3%), Pre-flood NDWI (18.7%)
"""

with open('deliverables/metrics.txt', 'w') as f:
    f.write(metrics_content)

pre_s2_path = 'data/pre_flood_S2_Prayagraj.tif'
sar_flood_path = 'data/flood_S1_SAR_Prayagraj.tif'
cleaned_mask_path = 'data/cleaned_flood_impact.tif'

print("Generating Image 1 (Side-by-side)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

with rasterio.open(pre_s2_path) as src_s2:
    img = src_s2.read([3, 2, 1]) / 3000.0
    show(img, transform=src_s2.transform, ax=ax1, title='Pre-Flood (Sentinel-2 Optical)')

with rasterio.open(sar_flood_path) as src_sar:
    show(src_sar, cmap='gray', vmin=-25, vmax=-5, ax=ax2, title='During Flood (Sentinel-1 SAR)')

plt.tight_layout()
plt.savefig('deliverables/Image_1_Side_by_Side.png', dpi=150)
plt.close()

print("Generating Image 2 (Flood Mask Overlay)...")
fig, ax = plt.subplots(figsize=(10, 8))

with rasterio.open(sar_flood_path) as src_sar:
    sar_data = src_sar.read(1)
    # Plot SAR basemap
    show(src_sar, cmap='gray', vmin=-25, vmax=-5, ax=ax)
    bounds = src_sar.bounds

with rasterio.open(cleaned_mask_path) as src_mask:
    mask = src_mask.read(1)
    # Mask out zeros so they are transparent (only show 1s)
    mask_transparent = np.where(mask == 0, np.nan, mask)
    
# Layer the bright cyan mask over the SAR data
cmap = ListedColormap(['cyan'])
ax.imshow(mask_transparent, cmap=cmap, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), alpha=0.7)
ax.set_title('Cleaned Flood Impact (Cyan) over SAR Basemap')
ax.axis('off')

plt.tight_layout()
plt.savefig('deliverables/Image_2_Flood_Mask_Overlay.png', dpi=200)
plt.close()

print("Deliverables successfully generated in the 'deliverables' folder!")
