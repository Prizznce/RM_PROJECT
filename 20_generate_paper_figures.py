"""
20_generate_paper_figures.py
============================
Generates the final statistical visualization figures for the research paper:
1. Confusion Matrix Heatmap
2. ROC Curve
3. Feature Importance Bar Chart

Uses the trained predictive_flood_best_model.joblib and 2026 testing data.
"""
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

# Style configuration
BG_COLOR = '#0d1117'
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': '#30363d'
})

output_dir = "deliverables"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data for 2026 Test Set
print("Preparing 2026 testing data...")
def load_and_resample(filepath, match_dataset, band=1):
    with rasterio.open(filepath) as src:
        resampled_data = np.empty(match_dataset.shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, band),
            destination=resampled_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=match_dataset.transform,
            dst_crs=match_dataset.crs,
            resampling=Resampling.nearest
        )
        return resampled_data.flatten()

year = '2026'
label_path = f"data/{year}/flood_mask_{year}.tif"
optical_path = f"data/{year}/prayagraj_preflood_S2_{year}.tif"
static_topo = "data/prayagraj_topo_features.tif"
static_dist = "data/prayagraj_distance_to_river.tif"
static_twi  = "data/prayagraj_twi.tif"
static_lulc = "data/prayagraj_LULC_ESA_WorldCover.tif"
static_soil = "data/prayagraj_soil_clay_pct.tif"

with rasterio.open(label_path) as src_label:
    labels = src_label.read(1).flatten()
    elevation = load_and_resample(static_topo, src_label, band=1)
    slope = load_and_resample(static_topo, src_label, band=2)
    distance = load_and_resample(static_dist, src_label, band=1)
    twi = load_and_resample(static_twi, src_label, band=1)
    lulc = load_and_resample(static_lulc, src_label, band=1)
    soil_clay = load_and_resample(static_soil, src_label, band=1)
    green = load_and_resample(optical_path, src_label, band=2)
    nir = load_and_resample(optical_path, src_label, band=4)
    ndwi = (green - nir) / (green + nir + 1e-8)

df = pd.DataFrame({
    'Elevation': elevation,
    'Slope': slope,
    'Distance_To_River': distance,
    'TWI': twi,
    'LULC': lulc,
    'Soil_Clay_Pct': soil_clay,
    'Pre_Flood_NDWI': ndwi,
    'Target': labels
})

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Keep the data balanced like we did in training
water = df[df['Target'] == 1]
land = df[df['Target'] == 0]
n = min(5000, len(water), len(land))
test_df = pd.concat([
    water.sample(n=n, random_state=42),
    land.sample(n=n, random_state=42)
])

feature_cols = ['Elevation', 'Slope', 'Distance_To_River', 'TWI', 'LULC', 'Soil_Clay_Pct', 'Pre_Flood_NDWI']
X_test = test_df[feature_cols]
y_test = test_df['Target']

print(f"Test data size: {len(X_test)} pixels")

# 2. Load the best model
model_path = 'predictive_flood_best_model.joblib'
print(f"Loading model: {model_path}")
rf_model = joblib.load(model_path)

# Predict
print("Predicting on test data...")
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# --- PLOT 1: Confusion Matrix Heatmap ---
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
fig1, ax1 = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Dry', 'Flood'], yticklabels=['Dry', 'Flood'],
            ax=ax1, annot_kws={"size": 14, "weight": "bold"})
ax1.set_title('Confusion Matrix - 2026 Test Set', fontsize=16, fontweight='bold', pad=15, color='white')
ax1.set_xlabel('Predicted Label', fontsize=14, color='white')
ax1.set_ylabel('True Label', fontsize=14, color='white')
# Adjust ticks
ax1.tick_params(colors='white', labelsize=12)

path1 = os.path.join(output_dir, 'fig_confusion_matrix.png')
fig1.savefig(path1, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
plt.close(fig1)
print(f"  Saved: {path1}")

# --- PLOT 2: ROC Curve ---
print("Generating ROC Curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots(figsize=(8, 8), facecolor=BG_COLOR)
ax2.plot(fpr, tpr, color='#ff4757', lw=3, label=f'Random Forest (AUC = {roc_auc:.4f})')
ax2.plot([0, 1], [0, 1], color='#8b949e', lw=2, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=14, color='white')
ax2.set_ylabel('True Positive Rate', fontsize=14, color='white')
ax2.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold', pad=15, color='white')
ax2.legend(loc="lower right", facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=12)
ax2.grid(True, color='#30363d', linestyle='-', alpha=0.5)

path2 = os.path.join(output_dir, 'fig_roc_curve.png')
fig2.savefig(path2, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
plt.close(fig2)
print(f"  Saved: {path2}")

# --- PLOT 3: Feature Importance Bar Chart ---
print("Generating Feature Importance Chart...")
importances = rf_model.feature_importances_
# Create a dataframe for seaborn
imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
imp_df = imp_df.sort_values(by='Importance', ascending=False)

fig3, ax3 = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis', ax=ax3)
ax3.set_title('Feature Importance (Random Forest)', fontsize=16, fontweight='bold', pad=15, color='white')
ax3.set_xlabel('Relative Importance', fontsize=14, color='white')
ax3.set_ylabel('Features', fontsize=14, color='white')
ax3.tick_params(colors='white', labelsize=12)
ax3.grid(True, axis='x', color='#30363d', linestyle='-', alpha=0.5)

path3 = os.path.join(output_dir, 'fig_feature_importance.png')
fig3.savefig(path3, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
plt.close(fig3)
print(f"  Saved: {path3}")

print("\nDone! All research figures generated successfully.")
