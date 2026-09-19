import rasterio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

print("Loading data for Machine Learning model training...")

# 1. Load Labels (Our Ground Truth from the pipeline)
with rasterio.open(r"data/cleaned_flood_impact.tif") as src:
    labels = src.read(1).flatten()
    
# 2. Load Features (SAR Backscatter)
with rasterio.open(r"data/flood_S1_SAR_Prayagraj.tif") as src:
    sar_data = src.read(1).flatten()

# 3. Load Features (Pre-flood Optical for NDWI)
with rasterio.open(r"data/pre_flood_S2_Prayagraj.tif") as src:
    # Band 2 = Green, Band 4 = NIR (based on our previous analysis of GEE script)
    green = src.read(2).astype(float).flatten()
    nir = src.read(4).astype(float).flatten()
    # Calculate NDWI
    ndwi = (green - nir) / (green + nir)

# 4. Build Dataset
print("Building DataFrame...")
df = pd.DataFrame({
    'SAR_Backscatter': sar_data,
    'Pre_Flood_NDWI': ndwi,
    'Label': labels
})

# Drop NoData / NaN values
print(f"Total pixels before cleaning: {len(df)}")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Also filter out areas where SAR backscatter is exactly 0 (commonly NoData in GEE export)
df = df[df['SAR_Backscatter'] != 0]

print(f"Valid pixels after cleaning: {len(df)}")

# 5. Sample Data
# Using the entire dataset might be too large and slow for local training.
# Let's sample a balanced dataset: up to 10,000 Water pixels, 10,000 Land pixels
water_pixels = df[df['Label'] == 1]
land_pixels = df[df['Label'] == 0]

# Ensure we don't sample more than available
n_samples = min(10000, len(water_pixels), len(land_pixels))
if n_samples == 0:
    print("Error: Not enough data to sample (one of the classes is empty).")
    exit()

print(f"Sampling {n_samples} pixels per class for training...")
water_sampled = water_pixels.sample(n=n_samples, random_state=42)
land_sampled = land_pixels.sample(n=n_samples, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([water_sampled, land_sampled]).sample(frac=1, random_state=42)

X = balanced_df[['SAR_Backscatter', 'Pre_Flood_NDWI']]
y = balanced_df['Label']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train the Model
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 8. Evaluate Model
print("\n--- Model Evaluation ---")
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Land (0)', 'Water (1)']))

# Display Feature Importances
importances = rf_model.feature_importances_
print("\nFeature Importances:")
print(f"SAR Backscatter: {importances[0]:.4f}")
print(f"Pre-flood NDWI:  {importances[1]:.4f}")

# 9. Save the Model
model_filename = 'flood_rf_model.joblib'
joblib.dump(rf_model, model_filename)
print(f"\nModel successfully saved as: {model_filename}")
print("You can now load this model in the future to classify new pixels without manual dB tuning!")
