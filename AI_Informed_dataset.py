"""
==========================================================
ESP32 Heater Data ML Pipeline - Stage A & Stage B
==========================================================

Purpose:
--------
This script processes CSV logs generated from ESP32 heater experiments
(both stabilized and dynamic sweeps), and builds predictive ML models
for:

1. Stage A: Predict BME temperature (room temperature) from sensors.
2. Stage B: Predict PWM (heater control) based on temperature predictions
   and environmental/context features.

Key Notes:
----------
- CSV files are collected from independent daily runs; train/test splits
  are run-aware to avoid leakage.
- Stage A predicts room temperature (BME sensor).
- Stage B predicts heater PWM based on predicted temperatures,
  environmental data (outside temp, humidity, pressure), and simulated occupancy/activity.
- Thermistor 6 is used for heater temperature; thermistor out represents outside temperature.
- Non-numeric features are excluded from numeric computations.
- Missing feature values are filled with column mean; missing targets are dropped.
- Verbose logging prints rows loaded, missing values, train/test shapes, Stage A MAE/R2, Stage B accuracy.
- Stage B PWM is binned to reflect greater sensitivity at low PWM values.
- Feature importances are printed for interpretability.

Requirements:
-------------
- pandas, numpy, scikit-learn
"""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# ==========================================================
# Step 1: Load all Dynamic_ and Stabilized_ CSV files
# ==========================================================
csv_directory_path = '.'  # Current directory
csv_file_pattern = os.path.join(csv_directory_path, '*.csv')
all_csv_files = glob.glob(csv_file_pattern)

# Only include files starting with Dynamic_ or Stabilized_
file_names = [f for f in all_csv_files
              if os.path.basename(f).startswith('Dynamic_') or
                 os.path.basename(f).startswith('Stabilized_')]

if not file_names:
    raise ValueError(f"No CSV files found matching the criteria in: {csv_directory_path}")

# Read all CSVs, keep source filename
df_list = []
for f in file_names:
    df = pd.read_csv(f)
    df['source_file'] = os.path.basename(f)
    df_list.append(df)
    print(f"Loaded {os.path.basename(f)}: Rows={df.shape[0]}, Columns={df.shape[1]}")

combined_df = pd.concat(df_list, ignore_index=True)
print(f"\nCombined DataFrame shape: {combined_df.shape}")

# Standardize column names to lowercase for consistency
combined_df.columns = [c.strip().lower() for c in combined_df.columns]
print(f"Columns after standardization: {list(combined_df.columns)}")

# ==========================================================
# Step 2: Handle missing values
# ==========================================================
# Features for Stage A: heater thermistor + outside temperature
feature_cols_stage_a = ['thrmstr6', 'thrmstrout']
target_col_stage_a = 'bme_temp'

# Fill missing features with column mean
for col in feature_cols_stage_a:
    if combined_df[col].isnull().any():
        mean_val = combined_df[col].mean()
        combined_df[col].fillna(mean_val, inplace=True)
        print(f"Column '{col}' had missing values. Filled with mean={mean_val:.2f}")

# Drop rows with missing Stage A target
if combined_df[target_col_stage_a].isnull().any():
    n_missing = combined_df[target_col_stage_a].isnull().sum()
    combined_df = combined_df.dropna(subset=[target_col_stage_a])
    print(f"Dropping {n_missing} rows with missing target '{target_col_stage_a}'")

# ==========================================================
# Step 3: Stage A - Predict BME Temperature (room temp)
# ==========================================================
X = combined_df[feature_cols_stage_a]
y = combined_df[target_col_stage_a]

# Train/test split by runs to avoid data leakage
unique_runs = combined_df['source_file'].unique()
train_runs, test_runs = train_test_split(unique_runs, test_size=0.265, random_state=42)

train_mask = combined_df['source_file'].isin(train_runs)
test_mask = combined_df['source_file'].isin(test_runs)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nStage A Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

# Train Random Forest Regressor for Stage A
print("\nTraining Stage A Random Forest Regressor...")
rf_stage_a = RandomForestRegressor(n_estimators=100, random_state=42)
rf_stage_a.fit(X_train, y_train)

# Evaluate Stage A
y_pred_a = rf_stage_a.predict(X_test)
mae_a = mean_absolute_error(y_test, y_pred_a)
r2_a = r2_score(y_test, y_pred_a)
print(f"Stage A Test MAE: {mae_a:.4f}, R2: {r2_a:.4f}")

# ==========================================================
# Step 4: Simulate occupancy/activity for Stage B
# ==========================================================
# Placeholder for activity/occupancy; can be replaced with real simulation
combined_df['activity_sim'] = np.random.randint(0, 2, size=combined_df.shape[0])

# ==========================================================
# Step 5: Stage B - Predict PWM control
# ==========================================================
# Bin PWM values for discrete classification
bins = [0, 128, 256, 512, 768, 1024]
labels = [0, 1, 2, 3, 4]
combined_df['pwm_bin'] = pd.cut(combined_df['pwm'], bins=bins, labels=labels, include_lowest=True)

# Drop rows with missing PWM bins
combined_df = combined_df.dropna(subset=['pwm_bin'])

# Stage B features: core physics + environmental context + occupancy
feature_cols_stage_b = [
    'thrmstr6',     # Heater thermistor
    'thrmstrout',   # Outside temperature
    'bme_temp',     # Predicted room temp
    'bme_hum',      # Relative humidity
    'bme_press',    # Atmospheric pressure
    'activity_sim'  # Simulated occupancy/activity
]

X_b = combined_df[feature_cols_stage_b]
y_b = combined_df['pwm_bin'].astype(int)

# Train/test split by runs
train_mask_b = combined_df['source_file'].isin(train_runs)
test_mask_b = combined_df['source_file'].isin(test_runs)

X_train_b, X_test_b = X_b[train_mask_b], X_b[test_mask_b]
y_train_b, y_test_b = y_b[train_mask_b], y_b[test_mask_b]

print(f"\nStage B dataset shape after dropping NaNs: {X_train_b.shape[0] + X_test_b.shape[0]}, Features={len(feature_cols_stage_b)}")

# Train Random Forest Classifier
print("\nTraining Stage B Random Forest Classifier (PWM as discrete control)...")
rf_stage_b = RandomForestClassifier(n_estimators=100, random_state=42)
rf_stage_b.fit(X_train_b, y_train_b)

# Evaluate Stage B
y_pred_b = rf_stage_b.predict(X_test_b)
acc_b = accuracy_score(y_test_b, y_pred_b)
print(f"Stage B Test Accuracy: {acc_b:.4f}")

# Optional: feature importance analysis for interpretability
importances = rf_stage_b.feature_importances_
print("\nStage B Feature Importances:")
for f, imp in zip(feature_cols_stage_b, importances):
    print(f"  {f}: {imp:.3f}")

print("\n==========================================================")
print("Pipeline Complete: Stage A predicts BME temperature, Stage B predicts PWM control")
print("==========================================================")
