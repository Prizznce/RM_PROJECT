import rasterio
import numpy as np

with rasterio.open(r"data/flood_S1_SAR_Prayagraj.tif") as src:
    flood_sar = src.read(1)
    
valid_vals = flood_sar[~np.isnan(flood_sar) & (flood_sar != 0)]
print(f"SAR Min:  {np.min(valid_vals):.4f}")
print(f"SAR Max:  {np.max(valid_vals):.4f}")
print(f"SAR Mean: {np.mean(valid_vals):.4f}")
print(f"SAR Median: {np.median(valid_vals):.4f}")