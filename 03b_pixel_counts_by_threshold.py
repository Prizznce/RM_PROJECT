import rasterio
import numpy as np

with rasterio.open(r"data/flood_S1_SAR_Prayagraj.tif") as src:
    sar = src.read(1)
    valid = sar[~np.isnan(sar) & (sar != 0)]

for t in [-16.0, -17.0, -18.0, -19.0, -22.49]:
    count = np.sum((sar < t) & valid)
    print(f"Threshold {t} dB -> {count} pixels")