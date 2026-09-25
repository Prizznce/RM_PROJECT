import joblib
import matplotlib.pyplot as plt
import numpy as np

print("Generating Feature Importance Charts...")

# 1. Load the Topographic Susceptibility Model
topo_model = joblib.load('predictive_flood_rf_model.joblib')
topo_features = ['Elevation', 'Slope', 'Distance_To_River', 'Pre_Flood_NDWI']
topo_importances = topo_model.feature_importances_ * 100

# 2. Load the Weather-Driven Forecasting Model
weather_model = joblib.load('predictive_flood_rf_model_with_rainfall.joblib')
weather_features = ['Elevation', 'Slope', 'Distance_To_River', 'Pre_Flood_NDWI', 'Rainfall_7d', 'Rainfall_30d']
weather_importances = weather_model.feature_importances_ * 100

# Sort the features by importance for better visualization
topo_indices = np.argsort(topo_importances)[::-1]
weather_indices = np.argsort(weather_importances)[::-1]

# Create a side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Topographic Model
axes[0].bar([topo_features[i] for i in topo_indices], [topo_importances[i] for i in topo_indices], color='teal')
axes[0].set_title('Topographic Susceptibility Model\n(Accuracy: 73.2% - Strong Generalization)', fontsize=14)
axes[0].set_ylabel('Importance (%)', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate([topo_importances[i] for i in topo_indices]):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

# Plot 2: Weather-Driven Model
axes[1].bar([weather_features[i] for i in weather_indices], [weather_importances[i] for i in weather_indices], color='darkred')
axes[1].set_title('Weather-Driven Forecasting Model\n(Accuracy: 49.9% - Overfitting Collapse)', fontsize=14)
axes[1].set_ylabel('Importance (%)', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate([weather_importances[i] for i in weather_indices]):
    axes[1].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()

# Save the figure
output_path = 'deliverables/feature_importance_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Chart successfully saved to {output_path}")
