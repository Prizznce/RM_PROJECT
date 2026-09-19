import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.seterr(invalid='ignore')

# UPDATE THIS PATH to where your SAR image is saved
sar_flood_path = r"data/flood_S1_SAR_Prayagraj.tif" # Change to your actual SAR file name/path

with rasterio.open(sar_flood_path) as src:
    sar_data = src.read(1)
    
    # 1. Apply the Threshold
    # We tell numpy: if a pixel is less than -16 dB, make it a 1 (Water). Otherwise, make it a 0 (Land).
    water_mask = np.where(sar_data < -16, 1, 0)
    
    # 2. Calculate the Area
    # Sentinel-1 resolution is 10m. Therefore, one pixel = 10m x 10m = 100 square meters.
    pixel_area_sq_meters = 100
    total_water_pixels = np.sum(water_mask == 1)
    
    total_water_sq_meters = total_water_pixels * pixel_area_sq_meters
    total_water_sq_km = total_water_sq_meters / 1_000_000 # Convert to sq km
    
print(f"Total water pixels detected: {total_water_pixels}")
print(f"Total water area during flood: {total_water_sq_km:.2f} square kilometers")

# 3. Visualize the Binary Map
# Create a custom color map: 0 (Land) = transparent/white, 1 (Water) = Blue
cmap = ListedColormap(['#eeeeee', 'blue'])

plt.figure(figsize=(10, 8))
plt.imshow(water_mask, cmap=cmap)
plt.title(f'Binary Flood Extent Map\nTotal Water Area: {total_water_sq_km:.2f} sq km')
plt.axis('off') # Hides the coordinate axes for a cleaner map
plt.show()