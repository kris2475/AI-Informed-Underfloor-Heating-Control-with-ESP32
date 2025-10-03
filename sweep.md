# Heater PWM Thermal Sweep Data Logger

This Arduino/ESP32 sketch drives a heater through a dual-phase PWM sweep to generate comprehensive datasets for machine learning models. Sensor data is collected, filtered, displayed in real-time, and logged to an SD card.

---

## Table of Contents

- [Purpose](#purpose)
- [Filtering](#filtering)
- [Phases](#phases)
- [Sensors & Hardware](#sensors--hardware)
- [Data Logging](#data-logging)
- [Configuration Constants](#configuration-constants)
- [Setup Sequence](#setup-sequence)
- [Main Loop](#main-loop)
- [Logging & Filtering](#logging--filtering)
- [OLED Display](#oled-display)
- [PWM Control](#pwm-control)

---

## Purpose

- Drive a heater using ESP32 LEDC PWM.
- Collect data from multiple thermistors, BME280, INA260, and RTC.
- Log stabilized and dynamic sweep datasets for machine learning.
- Provide real-time feedback via OLED.

---

## Filtering

- **Sample Averaging:** Reduces noise in thermistor ADC readings.
- **Exponential Moving Average (EMA):** Applies once per logging cycle to further smooth readings.
- **Filtered Values:** Stored in `filteredThermistorVals[]` for each thermistor.

---

## Phases

1. **Pre-Calibration**
   - Waits for the system to stabilize at ambient temperature.
   - Monitors temperature changes using BME280 and thermistors.
2. **Stabilized Sweep**
   - Heater is driven through a predefined PWM sweep (0 → max → 0).
   - Waits for temperature stability before advancing each PWM step.
3. **Dynamic Sweep**
   - Applies PWM levels in a randomized order.
   - Holds each level for a random duration (1–5 minutes) to capture transient thermal responses.

---

## Sensors & Hardware

| Component | Interface | Pins / Address |
|-----------|----------|----------------|
| Thermistors | ADC | 39, 34, 35, 32, 33, 25, 26 |
| BME280 | I2C | 0x77 |
| INA260 | I2C | 0x40 |
| RTC DS3231 | I2C | Default |
| SSD1306 OLED | I2C | 0x3C |
| Heater | LEDC PWM | Pin 15 |
| SD Card | SPI | CS=5 |

---

## Data Logging

- Logs in **CSV format** with buffered writes (60 entries per flush).
- Creates **two separate files**:
  - `Stabilized_YYYYMMDD_HHMM.csv`
  - `Dynamic_YYYYMMDD_HHMM.csv`
- CSV columns include:
  - Timestamp, PWM, Voltage, Current, Power
  - Thermistor temperatures
  - BME280 temperature, humidity, pressure

---

## Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `EMA_ALPHA` | 0.1 | EMA smoothing factor |
| `HEATER_MAX_PWM` | 1023 | Maximum PWM duty cycle |
| `PWM_FREQ` | 500 Hz | LEDC PWM frequency |
| `DATA_UPDATE_INTERVAL_MS` | 1000 | Logging and display update interval |
| `BUFFER_SIZE` | 60 | Number of buffered entries before SD write |
| `STABILITY_THRESHOLD_C` | 0.1°C | Temperature change threshold for stability |
| `CALIBRATION_STABILITY_WINDOW_MS` | 900000 | 15 min pre-calibration stability window |
| `PROTOCOL_STABILITY_WINDOW_MS` | 120000 | 2 min stability check per PWM step |
| `MAX_CALIBRATION_TIMEOUT_MS` | 7200000 | 2-hour failsafe timeout |

---

## Setup Sequence

1. Initialize Serial for debugging.
2. Initialize **I2C bus** and sensors:
   - BME280, INA260, DS3231 RTC, SSD1306 OLED
3. Initialize **SD card**.
4. Generate full PWM sweep (`FULL_PWM_SWEEP`) and a randomized dynamic sweep (`DYNAMIC_PWM_SWEEP`).
5. Configure LEDC PWM for the heater.
6. Perform **Pre-Calibration**:
   - Fill filtered thermistor values.
   - Wait until BME280 temperature is stable.
7. Create **stabilized data CSV file** and write headers.

---

## Main Loop (`loop()`)

- Handles **state machine**:
  1. `STATE_STABILIZED_SWEEP`
     - Waits for temperature stability before advancing PWM step.
     - Moves to dynamic sweep when complete.
  2. `STATE_DYNAMIC_SWEEP`
     - Holds each PWM level for a random duration (1–5 min).
     - Moves to `STATE_COMPLETE` after final step.
  3. `STATE_PRE_CALIBRATION` & `STATE_COMPLETE` handled elsewhere.
- Updates **OLED display** and logs data every second.

---

## Logging & Filtering

- Reads **thermistor ADC values** with averaging (`readThermistorADC_Averaged()`).
- Applies **EMA filtering** to smooth temperature readings.
- Converts ADC to temperature using **Steinhart–Hart equation**.
- Logs:
  - Voltage, current, power from INA260
  - Filtered thermistor temperatures
  - BME280 sensor readings
- Writes buffered entries to SD card when buffer is full.

---

## OLED Display

Displays in real-time:

- Current **date and time** from RTC.
- **System status**:
  - Calibrating
  - Heating / Stable (Stabilized Sweep)
  - Dynamic Sweep
  - Complete
- Current PWM duty cycle and hold times.
- Temperature deltas, BME280 temperature, outside thermistor.
- Heater **power**.

---

## PWM Control

- Configured via ESP32 LEDC:
  - **Frequency:** 500 Hz
  - **Resolution:** 10 bits
  - **Channel:** 0
  - **Pin:** 15
- `setPWM(duty)` updates heater output.
- `setupLEDCPWM()` initializes PWM timer and channel.

---

## CSV Output Example

```csv
Date,Time,PWM,Voltage (V),Current (A),Power (W),Thrmstr1,Thrmstr2,Thrmstr3,Thrmstr4,Thrmstr5,Thrmstr6,ThrmstrOut,BME_Temp,BME_Hum,BME_Press
2025-10-03,12:00:01,120,12.345,1.234,15.2,23.45,23.50,23.48,23.47,23.46,23.44,22.50,24.12,45.6,1013.2
