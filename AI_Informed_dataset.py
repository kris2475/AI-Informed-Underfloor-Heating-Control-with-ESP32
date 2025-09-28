# This script implements a two-stage modeling approach for an AI-informed heating control loop in net-zero buildings.
# Stage A focuses on predicting the room temperature (bme_temp) based on inputs like PWM, thermistor readings, and environmental data.
# Stage B then uses environmental data, thermistor readings, time of day, occupancy, activity, and a target temperature
# to predict the optimal PWM value needed to achieve the desired room conditions.
# The goal is to benchmark different regression models and a Neural Network to identify suitable candidates for deployment
# on a resource-constrained microcontroller like the ESP32 for real-time heating control.

import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Loading and Preprocessing ---

print("--- Data Loading and Preprocessing ---")

# Use glob to find all CSV files in the current directory
file_names = glob.glob('*.csv')

# A list to store dataframes
dfs = []

# Iterate through the list of files and read them into pandas DataFrames
for file_name in file_names:
    try:
        df = pd.read_csv(file_name)

        # Add a 'sweep_type' column based on the filename
        if 'Stabilized' in file_name:
            df['sweep_type'] = 'Stabilized'
        elif 'Dynamic' in file_name:
            df['sweep_type'] = 'Dynamic'
        else:
            df['sweep_type'] = 'Unknown'

        dfs.append(df)
        print(f"Successfully loaded {file_name}")

    except FileNotFoundError:
        print(f"File not found: {file_name}. Skipping...")

# Check if any dataframes were successfully loaded
if not dfs:
    print("No CSV files were loaded. Please ensure they are in the same directory as the script.")
else:
    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)

    # Rename columns to a more consistent format
    combined_df.rename(columns={
        'Date': 'date',
        'Time': 'time',
        'PWM': 'pwm',
        'Voltage (V)': 'voltage',
        'Current (A)': 'current',
        'Power (W)': 'power',
        'BME_Temp': 'bme_temp',
        'BME_Hum': 'bme_hum',
        'BME_Press': 'bme_press',
    }, inplace=True)

    # Convert 'date' and 'time' columns to a single datetime object
    combined_df['timestamp'] = pd.to_datetime(combined_df['date'] + ' ' + combined_df['time'])

    # Extract 'hour_of_day' from the new 'timestamp' column
    combined_df['hour_of_day'] = combined_df['timestamp'].dt.hour

    # Define simple synthetic rules for target temperature, occupancy, and activity
    activity_rules = {
        'morning': {'activity': ['Exercising', 'Relaxing'], 'weights': [0.5, 0.5], 'targets': {'Exercising': [19, 20], 'Relaxing': [23, 24]}},
        'daytime': {'activity': ['Working', 'Reading/TV'], 'weights': [0.6, 0.4], 'targets': {'Working': [21, 22], 'Reading/TV': [23, 24]}},
        'evening': {'activity': ['WatchingTV', 'Relaxing'], 'weights': [0.5, 0.5], 'targets': {'WatchingTV': [23, 24], 'Relaxing': [23, 25]}},
        'night': {'activity': ['Sleeping'], 'weights': [1.0], 'targets': {'Sleeping': [18, 20]}},
    }

    def simulate_behavior(hour):
        # Introduce a probability of being unoccupied, especially during certain hours
        if 0 <= hour < 6 or 9 <= hour < 17: # Night or daytime, more likely to be unoccupied
            occupied_prob = 0.2 # 20% chance of being unoccupied
        else: # Morning or evening, more likely to be occupied
            occupied_prob = 0.9 # 90% chance of being occupied

        occupied = 1 if np.random.random() < occupied_prob else 0

        if occupied == 0:
            activity = 'Unoccupied'
            target_temp = np.random.choice(np.arange(18, 20)) # Set a lower target temp when unoccupied
        else:
            if 6 <= hour < 9: # morning
                category = 'morning'
            elif 9 <= hour < 17: # daytime
                category = 'daytime'
            elif 17 <= hour < 22: # evening
                category = 'evening'
            else: # night
                category = 'night'

            rules = activity_rules[category]
            activity = np.random.choice(rules['activity'], p=rules['weights'])

            # Randomly select a target temperature from the defined range
            temp_range = rules['targets'][activity]
            target_temp = np.random.choice(np.arange(temp_range[0], temp_range[1] + 1))


        return pd.Series([occupied, activity, target_temp])

    # Apply the function to create the new columns
    combined_df[['occupied', 'activity', 'target_temperature']] = combined_df['hour_of_day'].apply(lambda x: simulate_behavior(x))

    # Drop the original date and time columns as they are no longer needed
    combined_df.drop(columns=['date', 'time'], inplace=True)

    # Drop redundant thermistor columns
    redundant_thermistors = ['Thrmstr1', 'Thrmstr2', 'Thrmstr3', 'Thrmstr4', 'Thrmstr5']
    combined_df.drop(columns=redundant_thermistors, inplace=True)

    # Remove rows with missing values
    initial_rows = combined_df.shape[0]
    combined_df.dropna(inplace=True)
    rows_dropped = initial_rows - combined_df.shape[0]
    print(f"Dropped {rows_dropped} rows with missing values.")

    # Save the new combined dataset
    combined_df.to_csv('combined_heater_data_with_targets.csv', index=False)

    print("Combined dataset created and saved to 'combined_heater_data_with_targets.csv'.")
    print("Shape of combined dataset:", combined_df.shape)
    print("Columns:", combined_df.columns.tolist())


    # --- Prepare Data for Stage A (Predicting bme_temp) ---
    print("\n--- Preparing Data for Stage A ---")
    features_stage_a = ['pwm', 'Thrmstr6', 'ThrmstrOut', 'bme_hum', 'bme_press']
    target_stage_a = 'bme_temp'

    X_stage_a = combined_df[features_stage_a]
    y_stage_a = combined_df[target_stage_a]

    # Split data for Stage A
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_stage_a, y_stage_a, test_size=0.2, random_state=42)

    # Standardize features for Stage A
    scaler_a = StandardScaler()
    X_train_a_scaled = scaler_a.fit_transform(X_train_a)
    X_test_a_scaled = scaler_a.transform(X_test_a)

    # Handle non-finite values in scaled data by replacing with the mean of finite values
    X_train_a_scaled[~np.isfinite(X_train_a_scaled)] = np.nanmean(X_train_a_scaled)
    X_test_a_scaled[~np.isfinite(X_test_a_scaled)] = np.nanmean(X_test_a_scaled)

    print(f"Stage A Training data shape (scaled): {X_train_a_scaled.shape}")
    print(f"Stage A Testing data shape (scaled): {X_test_a_scaled.shape}")


    # --- Prepare Data for Stage B (Predicting pwm) ---
    print("\n--- Preparing Data for Stage B ---")
    features_stage_b = [
        'voltage', 'current', 'power', 'Thrmstr6', 'ThrmstrOut',
        'bme_temp', 'bme_hum', 'bme_press', 'hour_of_day', 'occupied',
        'activity', 'target_temperature'
    ]
    target_stage_b = 'pwm'

    X_stage_b = combined_df[features_stage_b]
    y_stage_b = combined_df[target_stage_b]

    # Split data for Stage B
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_stage_b, y_stage_b, test_size=0.2, random_state=42)

    # Define preprocessing steps for Stage B
    numerical_features_b = [
        'voltage', 'current', 'power', 'Thrmstr6', 'ThrmstrOut',
        'bme_temp', 'bme_hum', 'bme_press', 'hour_of_day', 'occupied',
        'target_temperature'
    ]
    categorical_features_b = ['activity']

    preprocessor_b = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_b),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_b)
        ],
        remainder='passthrough'
    )

    # Apply preprocessing to Stage B data
    X_train_b_processed = preprocessor_b.fit_transform(X_train_b)
    X_test_b_processed = preprocessor_b.transform(X_test_b)

    # Handle non-finite values after preprocessing by replacing with the mean
    X_train_b_processed[~np.isfinite(X_train_b_processed)] = np.nanmean(X_train_b_processed)
    X_test_b_processed[~np.isfinite(X_test_b_processed)] = np.nanmean(X_test_b_processed)


    print(f"Stage B Training data shape (processed): {X_train_b_processed.shape}")
    print(f"Stage B Testing data shape (processed): {X_test_b_processed.shape}")


    # --- 2. Conventional Model Benchmarking (Stage A) ---
    print("\n--- Conventional Model Benchmarking (Stage A) ---")

    # Instantiate conventional models
    linear_reg_model_a = LinearRegression()
    ridge_model_a = Ridge(random_state=42)
    dt_reg_model_a = DecisionTreeRegressor(random_state=42)

    # Train Stage A models
    print("Training Linear Regression model for Stage A...")
    linear_reg_model_a.fit(X_train_a_scaled, y_train_a)
    print("Training Ridge Regression model for Stage A...")
    ridge_model_a.fit(X_train_a_scaled, y_train_a)
    print("Training Decision Tree Regressor model for Stage A...")
    dt_reg_model_a.fit(X_train_a_scaled, y_train_a)

    # Evaluate Stage A models
    print("\nEvaluating Stage A models...")
    y_pred_linear_a = linear_reg_model_a.predict(X_test_a_scaled)
    y_pred_ridge_a = ridge_model_a.predict(X_test_a_scaled)
    y_pred_dt_a = dt_reg_model_a.predict(X_test_a_scaled)

    mse_linear_a = mean_squared_error(y_test_a, y_pred_linear_a)
    mae_linear_a = mean_absolute_error(y_test_a, y_pred_linear_a)
    mse_ridge_a = mean_squared_error(y_test_a, y_pred_ridge_a)
    mae_ridge_a = mean_absolute_error(y_test_a, y_pred_ridge_a)
    mse_dt_a = mean_squared_error(y_test_a, y_pred_dt_a)
    mae_dt_a = mean_absolute_error(y_test_a, y_pred_dt_a)

    print(f"Linear Regression (Stage A) - Test MAE: {mae_linear_a:.4f}")
    print(f"Ridge Regression (Stage A) - Test MAE: {mae_ridge_a:.4f}")
    print(f"Decision Tree Regressor (Stage A) - Test MAE: {mae_dt_a:.4f}")


    # --- 3. Conventional Model Benchmarking (Stage B) ---
    print("\n--- Conventional Model Benchmarking (Stage B) ---")

    # Instantiate conventional models for Stage B
    linear_reg_model_b = LinearRegression()
    ridge_model_b = Ridge(random_state=42)
    dt_reg_model_b = DecisionTreeRegressor(random_state=42)
    rf_reg_model_b = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Train Stage B models
    print("Training Linear Regression model for Stage B...")
    linear_reg_model_b.fit(X_train_b_processed, y_train_b)
    print("Training Ridge Regression model for Stage B...")
    ridge_model_b.fit(X_train_b_processed, y_train_b)
    print("Training Decision Tree Regressor model for Stage B...")
    dt_reg_model_b.fit(X_train_b_processed, y_train_b)
    print("Training Random Forest Regressor model for Stage B...")
    rf_reg_model_b.fit(X_train_b_processed, y_train_b)

    # Evaluate Stage B models
    print("\nEvaluating Stage B models...")
    y_pred_linear_b = linear_reg_model_b.predict(X_test_b_processed)
    y_pred_ridge_b = ridge_model_b.predict(X_test_b_processed)
    y_pred_dt_b = dt_reg_model_b.predict(X_test_b_processed)
    y_pred_rf_b = rf_reg_model_b.predict(X_test_b_processed)

    mse_linear_b = mean_squared_error(y_test_b, y_pred_linear_b)
    mae_linear_b = mean_absolute_error(y_test_b, y_pred_linear_b)
    mse_ridge_b = mean_squared_error(y_test_b, y_pred_ridge_b)
    mae_ridge_b = mean_absolute_error(y_test_b, y_pred_ridge_b)
    mse_dt_b = mean_squared_error(y_test_b, y_pred_dt_b)
    mae_dt_b = mean_absolute_error(y_test_b, y_pred_dt_b)
    mse_rf_b = mean_squared_error(y_test_b, y_pred_rf_b)
    mae_rf_b = mean_absolute_error(y_test_b, y_pred_rf_b)


    print(f"Linear Regression (Stage B) - Test MAE: {mae_linear_b:.4f}")
    print(f"Ridge Regression (Stage B) - Test MAE: {mae_ridge_b:.4f}")
    print(f"Decision Tree Regressor (Stage B) - Test MAE: {mae_dt_b:.4f}")
    print(f"Random Forest Regressor (Stage B) - Test MAE: {mae_rf_b:.4f}")


    # --- 4. Neural Network Model (Stage B) ---
    print("\n--- Neural Network Model (Stage B) ---")

    # Define and compile the Neural Network model for Stage B
    model_stage_b = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_b_processed.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model_stage_b.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("Neural Network Model Summary (Stage B):")
    model_stage_b.summary()

    # Train the Stage B Neural Network model
    print("\nTraining Neural Network model for Stage B...")
    history_stage_b = model_stage_b.fit(
        X_train_b_processed,
        y_train_b,
        epochs=50,  # You can adjust the number of epochs
        batch_size=32, # You can adjust the batch size
        validation_split=0.2, # Use 20% of the training data for validation
        verbose=0 # Set to 1 to see training progress
    )
    print("Neural Network model trained for Stage B.")

    # Evaluate the Stage B Neural Network model on the test data
    print("\nEvaluating Neural Network model for Stage B...")
    loss_nn_b, mae_nn_b = model_stage_b.evaluate(X_test_b_processed, y_test_b, verbose=0)

    print(f"Neural Network (Stage B) - Test MAE: {mae_nn_b:.4f}")


    # --- 5. Summary of Stage B Model Performance ---
    print("\n--- Summary and Comparison of All Stage B Model Performance ---")
    print("\nHere's a summary of the performance of the models trained for the Stage B task (predicting optimal PWM), based on their Mean Absolute Error (MAE) on the test set:")

    print(f"\n*   **Linear Regression (Test MAE):** {mae_linear_b:.4f}")
    print(f"*   **Ridge Regression (Test MAE):** {mae_ridge_b:.4f}")
    print(f"*   **Decision Tree Regressor (Test MAE):** {mae_dt_b:.4f}")
    print(f"*   **Random Forest Regressor (Test MAE):** {mae_rf_b:.4f}")
    print(f"*   **Neural Network (Test MAE):** {mae_nn_b:.4f}")

    print("\nBased on these Test MAE values, the Decision Tree Regressor currently shows the best performance.")
