import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open(r"data/flood_S1_SAR_Prayagraj.tif") as src:
    sar = src.read(1)
    transform = src.transform
    pixel_res = abs(transform[0])
    pixel_area_sq_m = pixel_res ** 2

# Apply median filter
smoothed_sar = ndimage.median_filter(sar, size=5)
valid = ~np.isnan(sar) & (sar != 0)

# Test a range of thresholds from strict to loose
for thresh in [-16.0, -18.0, -20.0, -22.0]:
    raw_mask = (smoothed_sar < thresh) & valid
    
    # Label components to find the largest single water body size
    labeled, num_features = ndimage.label(raw_mask)
    if num_features > 0:
        sizes = ndimage.sum(raw_mask, labeled, range(1, num_features + 1))
        max_size = np.max(sizes)
        print(f"Threshold: {thresh} dB | Total raw pixels: {np.sum(raw_mask)} | Largest blob size (pixels): {max_size}")
    else:
        print(f"Threshold: {thresh} dB | No pixels found.")