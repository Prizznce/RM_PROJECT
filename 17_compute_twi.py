"""
17_compute_twi.py
==================
Computes Topographic Wetness Index (TWI) from the existing DEM.
TWI = ln(a / tan(b))
  a = upstream contributing area
  b = local slope in radians
"""
import rasterio
import numpy as np
from scipy import ndimage
import warnings

warnings.filterwarnings('ignore')

topo_path = "data/prayagraj_topo_features.tif"
output_path = "data/prayagraj_twi.tif"

print("Computing Topographic Wetness Index (TWI)...")

with rasterio.open(topo_path) as src:
    elevation = src.read(1).astype(np.float64)
    profile = src.profile
    pixel_size = abs(src.transform[0]) * 111320  # degrees to meters
    cell_area = pixel_size ** 2

print(f"  DEM shape: {elevation.shape}")
print(f"  Elevation range: {elevation[elevation > 0].min():.0f} - {elevation.max():.0f} m")
print(f"  Pixel size: ~{pixel_size:.1f} m")

# Valid data mask
valid = elevation > 0

# ---- Step 1: Compute slope from DEM directly ----
print("  Computing slope from elevation gradient...")
dy, dx = np.gradient(elevation, pixel_size)
slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
slope_rad = np.maximum(slope_rad, 0.001)  # avoid div by zero

# ---- Step 2: Flow accumulation (simplified D8) ----
print("  Computing flow accumulation...")
flow_acc = np.ones_like(elevation)

for iteration in range(10):
    padded = np.pad(flow_acc, 1, mode='constant', constant_values=0)
    padded_elev = np.pad(elevation, 1, mode='constant', constant_values=9999)
    
    new_acc = np.ones_like(elevation)
    
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r_start = 1 + dr
            c_start = 1 + dc
            neighbor_elev = padded_elev[r_start:r_start+elevation.shape[0],
                                        c_start:c_start+elevation.shape[1]]
            neighbor_flow = padded[r_start:r_start+elevation.shape[0],
                                   c_start:c_start+elevation.shape[1]]
            
            # Receive flow from higher neighbors
            higher = neighbor_elev > elevation
            new_acc += np.where(higher, neighbor_flow * 0.125, 0)
    
    flow_acc = new_acc

# Specific contributing area
specific_area = flow_acc * cell_area / pixel_size

# ---- Step 3: Compute TWI ----
print("  Computing TWI...")
twi = np.log(specific_area / np.tan(slope_rad))
twi = np.clip(twi, 0, 25)
twi[~valid] = 0

twi_valid = twi[valid]
print(f"  TWI range: {twi_valid.min():.2f} to {twi_valid.max():.2f}")
print(f"  TWI mean:  {twi_valid.mean():.2f}")

# ---- Save ----
profile.update(dtype=rasterio.float32, count=1, compress='lzw')
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(twi.astype(np.float32), 1)

print(f"  Saved: {output_path}")
