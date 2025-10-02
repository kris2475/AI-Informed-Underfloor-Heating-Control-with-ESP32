# 🔥 ML-Driven Thermal Control Data Logger (ESP32 & Python Pipeline)

A complete project for generating comprehensive machine learning datasets from a temperature-controlled heating element, followed by a Python pipeline to model the system's thermal dynamics and predict optimal heater control.

The system uses an **ESP32** to execute a two-phase PWM sweep protocol, collecting highly filtered data from multiple thermistors and environmental sensors. The collected data is then processed by a **Python script** to train a two-stage Random Forest model for predicting system state and control action.

## ⚙️ Part 1: ESP32 Firmware (Data Acquisition Protocol)

The firmware on the ESP32 drives the heater through a structured protocol and logs sensor data to an SD card with filtering and buffering.

### ✨ Key Firmware Features

* **Dual-Phase PWM Protocol:** Executes two distinct testing phases to capture different thermal behaviors:
    1.  **Stabilized Sweep:** Runs the heater through a predefined $0 \rightarrow \text{max} \rightarrow 0$ PWM cycle, waiting for the system temperature to **stabilize** at each level to capture steady-state thermodynamics.
    2.  **Dynamic Sweep:** Immediately follows the stabilized sweep with a randomized sequence of PWM levels, each held for a **random duration** (1-5 minutes), to capture transient thermal responses for robust ML training.
* **Advanced Sensor Filtering:** Thermistor readings are processed with a two-step noise reduction: **Sample Averaging** followed by an **Exponential Moving Average (EMA)** filter, ensuring clean, reliable data for the ML model.
* **Comprehensive Sensor Suite:**
    * **7 Thermistors (D39, D34, etc.):** Multiple points for temperature measurement.
    * **BME280:** Measures ambient **Temperature, Humidity, and Pressure**.
    * **INA260:** Measures **Voltage, Current, and Power** of the heater.
    * **DS3231 RTC:** Provides highly accurate **timestamps** for all log entries.
* **Efficient Data Logging:** Logs data to **two separate CSV files** (`Stabilized_...csv` and `Dynamic_...csv`) on an SD card, using **buffered writes** to improve performance and card longevity.
* **Real-time Diagnostics:** Uses an **SSD1306 OLED** to display the current protocol state, PWM duty cycle, temperature stability metrics, and heater power in real-time.

### 🔌 Hardware & Wiring Summary

| Component | ESP32 Pin (GPIO) | Interface | Notes |
| :--- | :--- | :--- | :--- |
| **Heater PWM Control** | `GPIO_NUM_15` | $\text{LEDC} \text{ PWM}$ | 10-bit resolution (0-1023) |
| **Thermistors (x7)** | `GPIO_NUM_39`, `34`, `35`, `32`, `33`, `25`, `26` | $\text{ADC}$ | Must be connected via voltage divider circuit |
| **INA260** | $\text{SDA}$ (`21`), $\text{SCL}$ (`22`) | $\text{I2C}$ | Address `0x40` |
| **BME280 / DS3231** | $\text{SDA}$ (`21`), $\text{SCL}$ (`22`) | $\text{I2C}$ | Shares bus with $\text{INA260}$ and $\text{OLED}$ |
| **SD Card CS** | `GPIO_NUM_5` | $\text{SPI}$ | Chip Select for $\text{SD}$ Card Module |

---

## 💻 Part 2: Python ML Pipeline (`ESP32_Heater_Pipeline.py`)

This script is the back-end data processor and model training environment, designed to turn the raw experimental logs into actionable predictive models.

### 🚀 Usage and Requirements

**Requirements:**
```bash
pip install pandas numpy scikit-learn
