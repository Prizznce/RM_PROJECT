"""
14_model_comparison.py
======================
Fair head-to-head comparison: Random Forest vs XGBoost
Same features, same data, same temporal hold-out.
Outputs a clean comparison table for the research paper.
"""
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd
import os
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, roc_auc_score, f1_score)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')


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


# -- 1. BUILD THE MULTI-YEAR DATASET -------------------------------------
print("=" * 60)
print("  MULTI-MODEL COMPARISON: Random Forest vs XGBoost")
print("=" * 60)

years = ['2019', '2021', '2022', '2024', '2025', '2026']
all_dfs = []

static_topo = "data/prayagraj_topo_features.tif"
static_dist = "data/prayagraj_distance_to_river.tif"
static_twi  = "data/prayagraj_twi.tif"
static_lulc = "data/prayagraj_LULC_ESA_WorldCover.tif"
static_soil = "data/prayagraj_soil_clay_pct.tif"

for year in years:
    label_path = f"data/{year}/flood_mask_{year}.tif"
    optical_path = f"data/{year}/prayagraj_preflood_S2_{year}.tif"

    if all(map(os.path.exists, [label_path, optical_path, static_topo, static_dist])):
        print(f"\n  Loading year {year}...")
        try:
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
                'Target': labels,
                'Year': year
            })
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            water = df[df['Target'] == 1]
            land = df[df['Target'] == 0]
            n = min(5000, len(water), len(land))
            if n > 0:
                event_df = pd.concat([
                    water.sample(n=n, random_state=42),
                    land.sample(n=n, random_state=42)
                ])
                all_dfs.append(event_df)
                print(f"  [OK] {len(event_df)} balanced pixels from {year}")
            else:
                print(f"  [--] Skipping {year}: not enough data")
        except Exception as e:
            print(f"  [!!] Error in {year}: {e}")
    else:
        print(f"  [--] Missing files for {year}")

combined_df = pd.concat(all_dfs).sample(frac=1, random_state=42)

# -- 2. TEMPORAL HOLD-OUT SPLIT -------------------------------------------
feature_cols = ['Elevation', 'Slope', 'Distance_To_River', 'TWI', 'LULC', 'Soil_Clay_Pct', 'Pre_Flood_NDWI']

train_df = combined_df[combined_df['Year'].isin(['2019', '2021', '2022', '2024', '2025'])]
test_df = combined_df[combined_df['Year'] == '2026']

X_train, y_train = train_df[feature_cols], train_df['Target']
X_test, y_test = test_df[feature_cols], test_df['Target']

print(f"\n  Train: {len(X_train):,} pixels (2019-2025)")
print(f"  Test:  {len(X_test):,} pixels (2026 only)")

# -- 3. TRAIN BOTH MODELS ------------------------------------------------
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=8,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'-' * 60}")
    print(f"  Training {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=['Dry (0)', 'Inundated (1)'])

    results[name] = {
        'accuracy': acc, 'auc': auc, 'f1': f1,
        'confusion_matrix': cm, 'report': report,
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob
    }

    print(f"\n  {name} Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    {cm}")
    print(f"\n{report}")

    # Feature importances
    importances = model.feature_importances_
    print("  Feature Importances:")
    for feat, imp in sorted(zip(feature_cols, importances),
                            key=lambda x: x[1], reverse=True):
        print(f"    {feat:<22s}: {imp:.4f}")

# -- 4. SIDE-BY-SIDE COMPARISON TABLE ------------------------------------
print(f"\n{'=' * 60}")
print("  FINAL COMPARISON TABLE")
print(f"{'=' * 60}")
print(f"  {'Metric':<20s} {'Random Forest':>15s} {'XGBoost':>15s}")
print(f"  {'-' * 50}")
print(f"  {'Accuracy':<20s} {results['Random Forest']['accuracy']:>14.2%} {results['XGBoost']['accuracy']:>14.2%}")
print(f"  {'ROC AUC':<20s} {results['Random Forest']['auc']:>14.4f} {results['XGBoost']['auc']:>14.4f}")
print(f"  {'F1 Score':<20s} {results['Random Forest']['f1']:>14.4f} {results['XGBoost']['f1']:>14.4f}")

# Determine winner
rf_auc = results['Random Forest']['auc']
xgb_auc = results['XGBoost']['auc']
winner = 'Random Forest' if rf_auc >= xgb_auc else 'XGBoost'
print(f"\n  * Best Model (by AUC): {winner}")

# -- 5. SAVE BEST MODEL --------------------------------------------------
best_model = results[winner]['model']
joblib.dump(best_model, 'predictive_flood_best_model.joblib')
print(f"\n  Best model saved as: predictive_flood_best_model.joblib")

# Also save comparison to metrics file
with open('deliverables/metrics.txt', 'a') as f:
    f.write('\n\n--- MODEL COMPARISON: RF vs XGBoost ---\n')
    f.write(f'Training: 50,000 pixels (2019-2025) | Testing: 10,000 pixels (2026)\n')
    f.write(f'Features: Elevation, Slope, Distance_To_River, Pre_Flood_NDWI\n\n')
    f.write(f'{"Metric":<20s} {"Random Forest":>15s} {"XGBoost":>15s}\n')
    f.write(f'{"-" * 50}\n')
    f.write(f'{"Accuracy":<20s} {results["Random Forest"]["accuracy"]:>14.2%} {results["XGBoost"]["accuracy"]:>14.2%}\n')
    f.write(f'{"ROC AUC":<20s} {results["Random Forest"]["auc"]:>14.4f} {results["XGBoost"]["auc"]:>14.4f}\n')
    f.write(f'{"F1 Score":<20s} {results["Random Forest"]["f1"]:>14.4f} {results["XGBoost"]["f1"]:>14.4f}\n')
    f.write(f'\nBest Model: {winner}\n')

print(f"\n  Results appended to deliverables/metrics.txt")
print(f"{'=' * 60}")
