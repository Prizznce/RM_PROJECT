import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Ignore numpy warnings (like comparing NaNs or divide-by-zero)
np.seterr(divide='ignore', invalid='ignore')

# 1. Get the Total Water from SAR (During Flood)
with rasterio.open(r"data/flood_S1_SAR_Prayagraj.tif") as src_sar:
    sar_data = src_sar.read(1)
    # 1 = Total Water, 0 = Land
    total_water_mask = np.where(sar_data < -16, 1, 0) 

# 2. Get the Normal River from Sentinel-2 (Pre-Flood)
# Our GEE script exported bands: B4(Red)=1, B3(Green)=2, B2(Blue)=3, B8(NIR)=4
with rasterio.open(r"data/pre_flood_S2_Prayagraj.tif") as src_s2:
    green = src_s2.read(2).astype(float)
    nir = src_s2.read(4).astype(float)
    
    # NDWI Formula: (Green - NIR) / (Green + NIR)
    ndwi = (green - nir) / (green + nir)
    
    # 1 = Normal River, 0 = Land (NDWI > 0 is standard for water)
    normal_water_mask = np.where(ndwi > 0, 1, 0)

# 3. Isolate the NEWLY Flooded Land
# Subtract the normal river from the total flood water
new_flood_mask = total_water_mask - normal_water_mask

# Clean up any anomalies (like shifting sandbanks that were wet in May but dry in Sept)
new_flood_mask = np.where(new_flood_mask > 0, 1, 0)

# 4. Calculate the Area of Just the Flooded Land
pixel_area_sq_meters = 100
flooded_pixels = np.sum(new_flood_mask == 1)
flooded_sq_km = (flooded_pixels * pixel_area_sq_meters) / 1_000_000

print(f"Total NEWLY Flooded Area: {flooded_sq_km:.2f} square kilometers")

# 5. Visualize the Impact
# 0 = Land (Light Grey), 1 = Flood (Red)
cmap = ListedColormap(['#eeeeee', 'red'])

plt.figure(figsize=(10, 8))
plt.imshow(new_flood_mask, cmap=cmap)
plt.title(f'Flood Impact Map: Prayagraj\nNewly Flooded Land: {flooded_sq_km:.2f} sq km')
plt.axis('off')
plt.show()