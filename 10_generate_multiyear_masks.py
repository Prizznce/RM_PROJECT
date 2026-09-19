import rasterio
import numpy as np
from scipy.ndimage import median_filter
from skimage import morphology
import os

def process_sar_to_mask(year):
    input_path = f"data/prayagraj_flooddate_S1_SAR_{year}.tif"
    output_path = f"data/flood_mask_{year}.tif"
    
    if not os.path.exists(input_path):
        print(f"File {input_path} not found. Skipping year {year}.")
        return

    print(f"\nProcessing SAR data for year: {year}")
    
    with rasterio.open(input_path) as src:
        # Read the VH band (typically band 1 in our export)
        sar_image = src.read(1)
        profile = src.profile

    print("1. Applying spatial median filter (size=5) to smooth speckle noise...")
    smoothed_sar = median_filter(sar_image, size=5)

    print("2. Thresholding at -16.0 dB...")
    # Pixels less than -16.0 dB are considered water (1), else land (0)
    binary_mask = (smoothed_sar < -16.0).astype(bool)

    print("3. Filtering noise (removing objects < 50 pixels)...")
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=50)

    # Convert back to uint8 for saving
    final_mask = cleaned_mask.astype(np.uint8)

    # Update profile for 1-band uint8 output
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(final_mask, 1)

    print(f"Successfully saved mask to {output_path}")

if __name__ == "__main__":
    years = ['2019', '2021', '2026']
    for year in years:
        process_sar_to_mask(year)
    
    print("\nMulti-year ground truth generation complete!")
