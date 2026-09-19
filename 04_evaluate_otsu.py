import rasterio
import numpy as np
from skimage.filters import threshold_otsu

np.seterr(invalid='ignore')

# 1. Load the flood-date SAR image
sar_path = r"data/flood_S1_SAR_Prayagraj.tif"

with rasterio.open(sar_path) as src:
    sar_data = src.read(1)
    
    # Sentinel-1 GEE export scale is 10m (10x10 = 100 sq meters per pixel)
    pixel_area_sq_m = 100

# Flatten the array and remove NoData values (NaN or typical NoData markers like 0 or -9999)
# Otsu's method will skew heavily if it tries to cluster NoData borders
valid_pixels = sar_data[~np.isnan(sar_data) & (sar_data != 0)]

# 2. Compute the optimal threshold using Otsu's method
otsu_thresh = threshold_otsu(valid_pixels)

# 3. Print out the chosen threshold to compare against the -16 dB baseline
print(f"Hardcoded Baseline Threshold: -16.00 dB")
print(f"Otsu's Calculated Threshold:  {otsu_thresh:.2f} dB")
print(f"Difference: {abs(otsu_thresh - (-16)):.2f} dB")

# 4. Apply the threshold (water has low backscatter, so it is strictly less than the threshold)
water_mask = (sar_data < otsu_thresh) & ~np.isnan(sar_data) & (sar_data != 0)

# 5. Recalculate the total water area in square kilometers
total_water_pixels = np.sum(water_mask)
total_water_sq_km = (total_water_pixels * pixel_area_sq_m) / 1_000_000

print(f"\nTotal Water Area (using Otsu): {total_water_sq_km:.2f} sq km")