ESP32 Heater ML Pipeline Report
1. Introduction
This report summarizes the development and evaluation of a machine learning (ML) pipeline designed to model and predict the behavior of an ESP32-controlled heating system. The pipeline processes experimental logs from both stabilized and dynamic sweeps to produce predictive models in two stages:
1. Stage A: Predicts the room temperature (BME sensor) from heater thermistor readings and outside temperature.
2. Stage B: Predicts heater PWM control based on room temperature predictions, heater thermistor, environmental sensors (BME humidity and pressure), and simulated activity/occupancy.

The approach aims to capture the underlying thermal physics of the system, account for environmental effects, and provide a foundation for intelligent heater control.
2. Data Collection & Processing
2.1 Source Data
• Files: Logs from 14 experimental runs, including both Dynamic_*.csv and Stabilized_*.csv.
• Sensors:
  o Thermistors: Thrmstr1–Thrmstr6, ThrmstrOut (outside temperature)
  o BME280: Temperature, humidity, pressure
  o INA260: Voltage, current, power
  o PWM heater control
• Data characteristics:
  o Number of rows: ~287,780 combined
  o Number of columns: 17 (thermistors, BME readings, PWM, voltage, current, power, source_file)
2.1.1 Stabilized vs Dynamic Sweeps
• Stabilized sweeps: Data captured after the system has reached thermal equilibrium at each PWM setpoint. These runs act as anchors for the model, providing reliable, low-noise measurements of steady-state heater and room temperatures. This is crucial for Stage A regression to accurately learn the baseline relationship.

• Dynamic sweeps: Data captured during transitions, when the heater is ramping up or down. These datasets capture the transient response of the system, including thermal inertia and lag effects.

• Importance in current models:
  o Stage A primarily benefits from stabilized sweeps.
  o Stage B uses both stabilized and dynamic sweeps to understand full system behavior.
  o Together, they provide both anchors (steady-state references) and transient dynamics.
2.2 Preprocessing
1. Column standardization: All column names converted to lowercase.
2. Missing values:
   o Features: Filled with column mean.
   o Targets: Dropped rows with missing BME temperature or PWM bins.
3. Run-aware splitting: Training/testing datasets split by experimental runs to avoid leakage.
3. Stage A: Room Temperature Prediction
Features:
• thrmstr6 (heater temp)
• thrmstrout (outside temp)

Model: Random Forest Regressor
Results:
• Test MAE: 0.706–0.884 °C
• R²: ~0.969–0.975

Interpretation: The model accurately predicts room temperature across runs, with strong generalization.
4. Stage B: Heater PWM Control Prediction
Features:
• thrmstr6
• thrmstrout
• bme_temp (Stage A prediction)
• bme_hum
• bme_press
• activity_sim (simulated occupancy)

Target: PWM bins [0–128, 128–256, 256–512, 512–768, 768–1024] mapped to [0–4]

Model: Random Forest Classifier
Accuracy: 0.65–0.75 depending on number of runs

Feature Importances:
• thrmstr6: ~0.24–0.27
• thrmstrout: ~0.18–0.22
• bme_temp: ~0.19–0.20
• bme_hum: ~0.19
• bme_press: ~0.15–0.17
• activity_sim: ~0.001

Interpretation: Heater and room temperatures dominate, while environmental features provide context. Occupancy currently contributes little as it is simulated.
5. Analysis & Discussion
Adding more runs increased dataset diversity but reduced Stage B accuracy (from ~75% to ~65%).

Why?
• New runs may contain conditions (different ambient temps, noise, non-ideal transitions) not present in earlier runs.
• The model is challenged to generalize across more variation.

This tradeoff is expected in ML: more data increases coverage but can reduce accuracy until model complexity, feature engineering, or temporal dynamics are improved.
6. Potential Improvements
• Use more advanced models (gradient boosting, deep learning).
• Add temporal dynamics (lag features, recurrent models).
• Replace random activity with real occupancy/setpoints.
• Tune hyperparameters for Random Forest.
• Consider finer PWM resolution if sufficient data supports it.
7. Conclusion
Stage A: High accuracy (R² ~0.97) for room temperature prediction.
Stage B: 65–75% accuracy for PWM control depending on dataset size.

The pipeline is physics-informed and interpretable. Accuracy dips with more data highlight the need for improved modeling techniques, but the system provides a solid foundation for intelligent ESP32 heater control.
