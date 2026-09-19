import rasterio
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects

np.seterr(invalid='ignore')

FLOOD_SAR_PATH = r"data/flood_S1_SAR_Prayagraj.tif"
OUT_CLEANED_PATH = r"data/cleaned_flood_impact.tif"

# Minimum Mapping Unit (MMU) threshold: 50 pixels = 5,000 sq meters (0.5 hectares)
MIN_PIXEL_SIZE = 50  

def main():
    print("Running Final Validated SAR Flood Pipeline...")

    with rasterio.open(FLOOD_SAR_PATH) as src:
        flood_sar = src.read(1)
        meta = src.meta.copy()
        
        # Sentinel-1 GEE export scale is 10m (10x10 = 100 sq meters per pixel)
        pixel_area_sq_m = 100

    # 1. Apply size-5 median filter for speckle smoothing
    print("-> Applying speckle smoothing (Median Filter size=5)...")
    smoothed_sar = ndimage.median_filter(flood_sar, size=5)

    # 2. Use -16.0 dB threshold (proven to capture the massive 319k pixel river body)
    optimal_thresh = -16.0
    print(f"-> Using Target Water Threshold: {optimal_thresh} dB")

    # Raw water mask
    valid_mask = ~np.isnan(flood_sar) & (flood_sar != 0)
    flood_water_raw = (smoothed_sar < optimal_thresh) & valid_mask
    initial_pixel_count = np.sum(flood_water_raw)
    print(f"-> Raw SAR water pixels found: {initial_pixel_count}")

    # 3. Apply Robust MMU Filter via scikit-image
    print("-> Applying Minimum Mapping Unit (MMU) spatial cleanup...")
    clean_mask = remove_small_objects(flood_water_raw, min_size=MIN_PIXEL_SIZE)
    
    final_water_pixels = np.sum(clean_mask)
    removed_blobs_approx = initial_pixel_count - final_water_pixels

    # 4. Calculate final area in sq km
    flood_sq_km = (final_water_pixels * pixel_area_sq_m) / 1_000_000

    print(f"\n==============================================")
    print(f"      FINAL FLOOD EXTENT RESULTS              ")
    print(f"==============================================")
    print(f"Remaining Water Pixels: {final_water_pixels}")
    print(f"Final Inundated Area:   {flood_sq_km:.2f} sq km")
    print(f"==============================================\n")

    # 5. Export Cleaned GeoTIFF for QGIS
    meta.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(OUT_CLEANED_PATH, 'w', **meta) as dst:
        dst.write(clean_mask.astype(rasterio.uint8), 1)

    print(f"Successfully exported: {OUT_CLEANED_PATH}")

if __name__ == "__main__":
    main()