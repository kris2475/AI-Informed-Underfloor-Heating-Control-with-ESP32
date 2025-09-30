ESP32 Heater ML Pipeline

This repository contains the full machine learning pipeline for modeling and predicting the behavior of an ESP32-controlled heating system.
The pipeline processes experimental logs (both stabilized sweeps and dynamic sweeps) to train models that:

Stage A: Predict the room temperature (BME sensor) from heater thermistor and outside temperature.

Stage B: Predict heater PWM control decisions based on Stage A predictions, environmental sensors, and simulated activity.

📂 Project Structure
.
├── data/                  # CSV log files from experiments
│   ├── Dynamic_*.csv
│   ├── Stabilized_*.csv
├── ESP32_Heater_Pipeline.py   # Full Python pipeline
├── ESP32_Heater_ML_Report.docx  # Detailed report with analysis
├── README.md

⚙️ Requirements

Install the required Python libraries:

pip install pandas numpy scikit-learn

🚀 Usage

Place your CSV logs inside the data/ directory.
Filenames should begin with Dynamic_ or Stabilized_.

Run the pipeline:

python ESP32_Heater_Pipeline.py


The script will:

Load and merge all log files

Preprocess missing values

Train Stage A (Random Forest Regressor) to predict room temperature

Train Stage B (Random Forest Classifier) to predict heater PWM control

Print metrics, feature importance, and summaries

📊 Results

Stage A:

Test MAE ≈ 0.7 °C

R² ≈ 0.97 (excellent fit)

Stage B:

Accuracy ≈ 65–75% depending on dataset size and sweeps used

Feature importance highlights strong dependence on heater thermistor, room temperature, and environmental context

🔍 Key Insights

Stabilized sweeps anchor the steady-state thermal relationships.

Dynamic sweeps capture transient heating/cooling trajectories.

Stage A provides a physics-informed foundation by isolating thermal dynamics.

Stage B adds control logic, predicting PWM with environmental context.

Adding more sweeps can reduce accuracy due to higher run-to-run variability — indicating that additional feature engineering or temporal modeling may be needed.

📈 Next Steps

Incorporate temporal features (lagged inputs, moving averages).

Replace activity_sim with real occupancy-driven setpoints.

Explore finer PWM resolution or a regression approach.

Add hyperparameter tuning and ensemble methods.

📝 Documentation

See the full technical write-up here:
📄 ESP32 Heater ML Pipeline Report
