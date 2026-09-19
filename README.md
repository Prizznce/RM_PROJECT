# Flood Impact Mapping Pipeline (Sentinel-1 & Sentinel-2)

This repository contains a remote sensing research pipeline designed to accurately map flood inundation and calculate newly flooded land area using a hybrid approach with **Sentinel-1 SAR** (Synthetic Aperture Radar) and **Sentinel-2 Optical** imagery.

## Overview
The goal of this project is to go beyond simple water detection and isolate the **actual impact** of a flood event (i.e., newly submerged land) by separating it from the normal, pre-existing water bodies (rivers, lakes, etc.).

We achieved this by leveraging the cloud-penetrating capabilities of SAR for the during-flood analysis, combined with high-resolution pre-flood optical data to establish a baseline.

## What Has Been Done So Far

### 1. Data Acquisition & Preprocessing
* Built a Google Earth Engine (GEE) script (`00_gee_data_pull.js`) to precisely filter, clip, and export Sentinel-1 (VH band) and Sentinel-2 (Multispectral) imagery for the Prayagraj study area during the 2019 flood event.
* Addressed NoData edge cases and implemented correct pixel-to-area georeferencing logic based on the 10-meter export scale.

### 2. Exploratory Data Analysis & Thresholding
* **Visual Baseline:** Implemented RGB True Color rendering of pre-flood optical data alongside grayscale SAR data (`01_visualize_inputs.py`).
* **Statistical Inspection:** Analyzed the distribution of SAR backscatter values (`02_inspect_sar_stats.py`) to inform threshold selection.
* **Threshold Calibration:** Systematically tested various dB thresholds and compared our empirical threshold (-16.0 dB) against algorithmic baselines like Otsu's method (`03_test_thresholds.py`, `04_evaluate_otsu.py`).

### 3. Advanced Geospatial Filtering Pipeline
To prevent radar speckle and tiny depressions from skewing the area calculations, we engineered a robust filtering pipeline (`06_clean_flood_pipeline.py`):
* **Median Filtering:** Applied a size-5 median spatial filter to smooth out high-frequency radar speckle noise.
* **Minimum Mapping Unit (MMU):** Utilized morphological object removal to eliminate disconnected "puddles" smaller than 5,000 sq meters, ensuring only significant contiguous water bodies are mapped.

### 4. True Impact Calculation
* Generated the normal pre-flood river baseline using the **Normalized Difference Water Index (NDWI)** derived from Sentinel-2's Green and NIR bands.
* Subtracted the normal pre-flood river mask from the cleaned SAR total-flood mask.
* Successfully isolated and visualized the **Newly Flooded Area**, calculating the final impact metric in square kilometers (`08_calculate_actual_impact.py`).

---

## Where Are We Heading Next? (Future Scope)

Having successfully established a robust methodology for isolating the newly flooded extent, the next phases of this research will focus on practical impact assessment and automation:

1. **Infrastructure & Demographic Impact Analysis**
   * **Overlaying Datasets:** Integrating the final flood mask with OpenStreetMap (OSM) building/road footprints and population density datasets (e.g., WorldPop).
   * **Outcome:** Quantifying the exact number of buildings submerged, kilometers of road damaged, and the estimated population displaced by the event.

2. **Adaptive / Machine Learning Thresholds**
   * **Current Limitation:** Using a global threshold (like -16.0 dB) can struggle across highly heterogeneous terrain (e.g., urban centers vs. agricultural fields).
   * **Next Step:** Implementing localized adaptive thresholding or training a lightweight Random Forest pixel classifier to automatically determine the water/land boundary without manual dB tuning.

3. **Time-Series Progression & Recession Modeling**
   * Expanding the GEE data pull to acquire an image time-series rather than a single snapshot.
   * Mapping the temporal progression of the flood (how fast it peaked and how slowly the water receded), which is critical for epidemiological risk assessment (e.g., stagnant water mosquito breeding).

4. **WebGIS / Dashboard Deployment**
   * Packaging the final outputs into an interactive dashboard (using Folium, Streamlit, or Google Earth Engine Apps) to allow stakeholders and disaster management authorities to dynamically explore the impact zones.