import rasterio
import matplotlib.pyplot as plt
import numpy as np

year = '2026'
input_path = f"data/flood_risk_map_{year}.tif"
output_path = f"deliverables/flood_risk_heatmap_{year}.png"

print(f"Loading {input_path} to render as an image...")

with rasterio.open(input_path) as src:
    # Read the probability data (0 to 100)
    data = src.read(1)
    
    # Mask out exactly 0 values (completely dry land) so they appear transparent/white
    data_masked = np.ma.masked_where(data < 5, data) # Hide anything under 5% risk

    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Use 'YlOrRd' (Yellow to Orange to Red) colormap
    # vmin=0, vmax=100 ensures the scale is locked to 0-100%
    im = plt.imshow(data_masked, cmap='YlOrRd', vmin=0, vmax=100)
    
    # Add a colorbar
    cbar = plt.colorbar(im, label='Flood Probability (%)', fraction=0.046, pad=0.04)
    
    plt.title(f"Predictive Flood Susceptibility Map - {year}")
    plt.axis('off') # Hide the axes ticks
    
    # Save the image
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False)
    print(f"Heatmap successfully rendered and saved to: {output_path}")
