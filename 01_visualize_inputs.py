import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# Update these paths if your folder structure is different
pre_s2_path = 'data/pre_flood_S2_Prayagraj.tif'
sar_flood_path = 'data/flood_S1_SAR_Prayagraj.tif'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Plot Pre-flood Optical
with rasterio.open(pre_s2_path) as src_s2:
    # S2 was exported with B4(Red), B3(Green), B2(Blue). Rasterio reads them as bands 3, 2, 1 for RGB.
    # Normalizing values (0-3000 typical for S2 SR) for display
    img = src_s2.read([3, 2, 1]) / 3000.0 
    show(img, transform=src_s2.transform, ax=ax1, title='Pre-Flood (Sentinel-2 Optical)')

# Plot During-flood SAR
with rasterio.open(sar_flood_path) as src_sar:
    # SAR is a single band (VH). Displaying in grayscale.
    show(src_sar, cmap='gray', vmin=-25, vmax=-5, ax=ax2, title='During Flood (Sentinel-1 SAR)')

plt.show()