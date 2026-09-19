import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the newly saved, cleaned GeoTIFF
with rasterio.open(r"data/cleaned_flood_impact.tif") as src:
    cleaned_mask = src.read(1)

# Visualize it
cmap = ListedColormap(['#eeeeee', 'red'])

plt.figure(figsize=(10, 8))
plt.imshow(cleaned_mask, cmap=cmap)
plt.title('Cleaned Flood Impact Map\nNotice the reduced radar noise (red dots)')
plt.axis('off')
plt.show()