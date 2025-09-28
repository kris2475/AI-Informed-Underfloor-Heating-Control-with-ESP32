/*
 * Purpose:
 * This sketch drives a heater through a dual-phase PWM sweep
 * to generate a comprehensive dataset for machine learning models.
 * Data is collected from a suite of sensors and logged to an SD card.
 *
 * Filtering:
 * To handle noise and sudden spikes in thermistor readings, a sample
 * averaging technique is first applied to the raw ADC readings. The
 * resulting averaged value is then passed to an Exponential Moving
 * Average (EMA) filter. The filter's state (the previous filtered value)
 * is stored in a global array. The filter is implemented as a "one-shot"
 * filter, meaning it is applied once per data logging cycle within the
 * logData() function, which guarantees a consistent and reliable
 * application of the filter for each sensor.
 *
 * Phases:
 * 1. Pre-Calibration: Waits for the system to stabilize at ambient.
 *
 * 2. Stabilized Sweep: Heats the system through a predefined PWM sweep
 * (0 -> max -> 0). At each step, it waits for temperature stability
 * before moving to the next level.
 *
 * 3. Dynamic Sweep: Immediately follows the stabilized sweep. Applies
 * the same PWM levels in a randomized order, holding each for a
 * random duration (1-5 minutes) to capture transient thermal responses.
 *
 * Sensors & Hardware:
 * - Thermistors (D39, D34, D35, D32, D33, D25, D26)
 * - BME280 (temperature, humidity, pressure)
 * - INA260 (voltage, current, power)
 * - RTC DS3231 for accurate timestamps
 * - SSD1306 OLED display for real-time status
 * - ESP32 LEDC PWM on pin 15
 *
 * Data Logging:
 * - Buffered writes to SD card (CSV format) for efficiency.
 * - The sketch now creates two separate files for stabilized and dynamic data.
 *
 */

#include <Wire.h>
#include "driver/ledc.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <RTClib.h>
#include <Adafruit_INA260.h>
#include <SD.h>
#include <Adafruit_BME280.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm> // For std::shuffle
#include <random>    // For std::mt19937
#include <esp_random.h> // For esp_random()

// --- Hardware and I2C Addresses ---
#define INA260_ADDR 0x40
#define BME280_ADDR 0x77
#define SCREEN_ADDRESS 0x3C
#define HEATER_PWM_PIN 15
#define SD_CS_PIN 5

// --- Instantiate Sensor Objects ---
Adafruit_SSD1306 display(128, 64, &Wire, -1);
RTC_DS3231 rtc;
Adafruit_INA260 ina260 = Adafruit_INA260();
Adafruit_BME280 bme;

// --- Thermistors ---
const int TEMP_SENSOR_PINS[] = {39, 34, 35, 32, 33, 25, 26};
const int NUM_THERMISTORS = sizeof(TEMP_SENSOR_PINS) / sizeof(TEMP_SENSOR_PINS[0]);
const char* THERMISTOR_LABELS[] = {"Thrmstr1", "Thrmstr2", "Thrmstr3", "Thrmstr4", "Thrmstr5", "Thrmstr6", "ThrmstrOut"};
#define THERMISTOR_NOMINAL 10000
#define TEMPERATURE_NOMINAL 25
#define B_COEFFICIENT 3950
#define SERIES_RESISTOR 10000
#define ADC_MAX_VALUE 4095
const float EMA_ALPHA = 0.1;

// --- PWM and Timing Constants ---
const int PWM_FREQ = 500;
const int HEATER_PWM_CHANNEL = 0;
const int HEATER_PWM_TIMER = 0;
const int HEATER_PWM_RESOLUTION = 10;
const int HEATER_MAX_PWM = 1023;
const unsigned long DATA_UPDATE_INTERVAL_MS = 1000;
const int BUFFER_SIZE = 60;

// --- PWM Values for the Sweep ---
const int INCREASING_PWM_VALUES[] = {
  0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40,
  50, 60, 70, 80, 100, 120, 140, 160, 200, 240, 280, 320, 400, 480, 560, 640,
  720, 800, 880, 960, 1020
};
const int NUM_INCREASING_PWM_LEVELS = sizeof(INCREASING_PWM_VALUES) / sizeof(INCREASING_PWM_VALUES[0]);

int FULL_PWM_SWEEP[NUM_INCREASING_PWM_LEVELS * 2 - 1];
const int NUM_PWM_LEVELS = NUM_INCREASING_PWM_LEVELS * 2 - 1;

int DYNAMIC_PWM_SWEEP[NUM_PWM_LEVELS];

// --- State Machine and Timing Variables ---
enum ProtocolState {
  STATE_PRE_CALIBRATION,
  STATE_STABILIZED_SWEEP,
  STATE_DYNAMIC_SWEEP,
  STATE_COMPLETE
};

ProtocolState currentState = STATE_PRE_CALIBRATION;
unsigned long testStartTime = 0;
unsigned long lastLogTime = 0;
int currentPWMIndex = 0;
unsigned long calibrationStartTime = 0;
unsigned long stableStartTime = 0;
float lastStabilityTemp = 0.0;
unsigned long lastStabilityCheckTime = 0;
bool isStable = false;
float lastTempDelta = 0.0;

// --- SD Card Buffering and File Variables ---
String dataBuffer[BUFFER_SIZE];
int bufferIndex = 0;
File stabilizedDataFile;
File dynamicDataFile;
char stabilizedFileName[30];
char dynamicFileName[30];

// --- Stabilization Protocol Variables ---
const float STABILITY_THRESHOLD_C = 0.1;
const unsigned long CALIBRATION_STABILITY_WINDOW_MS = 900000;    // 15 min
const unsigned long PROTOCOL_STABILITY_WINDOW_MS = 120000;     // 2 min
const unsigned long MAX_CALIBRATION_TIMEOUT_MS = 7200000;      // 2 hours failsafe

// --- Dynamic Sweep Variables ---
unsigned long dynamicHoldTime = 0;
unsigned long dynamicStepStartTime = 0;

// Filtered values for each thermistor
float filteredThermistorVals[NUM_THERMISTORS];
bool initialReadingsTaken = false;

// --- Function Prototypes ---
void setupLEDCPWM();
void updateDisplay();
void writeLogEntry(const String& logEntry, File& file);
void initializeSDFile();
float readThermistorTemp(int pin);
void performPreCalibration();
void setPWM(int duty);
void logData();
void setupStabilizedFile();
void setupDynamicFile();
float applyEMA(float newReading, float previousFilteredValue, float alpha);
float readThermistorADC_Averaged(int pin);


void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
    for (;;);
  }

  if (!rtc.begin()) {
    Serial.println(F("Couldn't find RTC"));
    for (;;);
  }
  if (rtc.lostPower()) {
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  }

  if (!ina260.begin(INA260_ADDR)) {
    Serial.println("Couldn't find INA260 chip");
    for (;;);
  }
  ina260.setMode(INA260_MODE_CONTINUOUS);
  ina260.setAveragingCount(INA260_COUNT_1024);

  if (!bme.begin(BME280_ADDR)) {
    Serial.println("Couldn't find BME280 chip");
    for (;;);
  }

  if (!SD.begin(SD_CS_PIN)) {
    Serial.println(F("SD Card initialization failed!"));
    while (1);
  }

  // Generate the full PWM sweep and a shuffled version for the dynamic test
  for (int i = 0; i < NUM_INCREASING_PWM_LEVELS; i++) {
    FULL_PWM_SWEEP[i] = INCREASING_PWM_VALUES[i];
    DYNAMIC_PWM_SWEEP[i] = INCREASING_PWM_VALUES[i];
  }
  for (int i = 1; i < NUM_INCREASING_PWM_LEVELS; i++) {
    FULL_PWM_SWEEP[NUM_INCREASING_PWM_LEVELS + i - 1] = INCREASING_PWM_VALUES[NUM_INCREASING_PWM_LEVELS - 1 - i];
    DYNAMIC_PWM_SWEEP[NUM_INCREASING_PWM_LEVELS + i - 1] = INCREASING_PWM_VALUES[NUM_INCREASING_PWM_LEVELS - 1 - i];
  }

  // Use a modern shuffle function with a high-quality random source
  std::mt19937 g(esp_random());
  std::shuffle(DYNAMIC_PWM_SWEEP, DYNAMIC_PWM_SWEEP + NUM_PWM_LEVELS, g);

  setupLEDCPWM();
  performPreCalibration();
  setupStabilizedFile();
  testStartTime = millis();
  setPWM(FULL_PWM_SWEEP[currentPWMIndex]);
  lastStabilityTemp = bme.readTemperature();
  lastStabilityCheckTime = millis();
  currentState = STATE_STABILIZED_SWEEP;
}

void loop() {
  unsigned long currentTime = millis();

  switch (currentState) {
    case STATE_STABILIZED_SWEEP:
      // Wait for stabilization before changing PWM level
      if (currentTime - lastStabilityCheckTime >= PROTOCOL_STABILITY_WINDOW_MS) {
        lastStabilityCheckTime = currentTime;
        float currentTemp = bme.readTemperature();
        lastTempDelta = abs(currentTemp - lastStabilityTemp);

        if (lastTempDelta < STABILITY_THRESHOLD_C) {
          isStable = true;
          currentPWMIndex++;

          if (currentPWMIndex < NUM_PWM_LEVELS) {
            setPWM(FULL_PWM_SWEEP[currentPWMIndex]);
          } else {
            // Stabilized sweep complete, move to dynamic sweep
            setPWM(0);
            delay(1000); // Wait 1 sec for logging
            setupDynamicFile();
            currentPWMIndex = 0;
            dynamicHoldTime = random(60000, 300001); // 1-5 min
            dynamicStepStartTime = millis();
            setPWM(DYNAMIC_PWM_SWEEP[currentPWMIndex]);
            currentState = STATE_DYNAMIC_SWEEP;
          }
        } else {
          isStable = false;
        }
        lastStabilityTemp = currentTemp;
      }
      break;

    case STATE_DYNAMIC_SWEEP:
      // Change PWM level after a random duration
      if (currentTime - dynamicStepStartTime >= dynamicHoldTime) {
        currentPWMIndex++;
        if (currentPWMIndex >= NUM_PWM_LEVELS) {
          setPWM(0);
          currentState = STATE_COMPLETE;

          // Flush any remaining data from buffer and close files
          for (int i = 0; i < bufferIndex; i++) {
            dynamicDataFile.println(dataBuffer[i]);
          }
          dynamicDataFile.close();
          stabilizedDataFile.close();

          while (true) {
            // End of test loop, display final message
            display.clearDisplay();
            display.setCursor(0, 0);
            display.setTextSize(1);
            display.setTextColor(SSD1306_WHITE);
            display.println("TEST COMPLETE");
            display.setTextSize(2);
            display.setCursor(0, 16);
            display.println("SAFE TO");
            display.setCursor(0, 32);
            display.println("REMOVE SD");
            display.setCursor(0, 48);
            display.println("CARD");
            display.display();
            delay(1000);
          }
        } else {
          setPWM(DYNAMIC_PWM_SWEEP[currentPWMIndex]);
          dynamicHoldTime = random(60000, 300001);
          dynamicStepStartTime = currentTime;
        }
      }
      break;

    case STATE_PRE_CALIBRATION:
    case STATE_COMPLETE:
      // Handled by other functions or does nothing
      break;
  }

  // Log data and update display every second
  if (currentTime - lastLogTime >= DATA_UPDATE_INTERVAL_MS) {
    lastLogTime = currentTime;
    logData();
    updateDisplay();
  }
}

// --- Helper Functions ---

void setupStabilizedFile() {
  DateTime now = rtc.now();
  sprintf(stabilizedFileName, "/Stabilized_%04d%02d%02d_%02d%02d.csv", now.year(), now.month(), now.day(), now.hour(), now.minute());
  stabilizedDataFile = SD.open(stabilizedFileName, FILE_WRITE);
  if (!stabilizedDataFile) { Serial.println("Failed to open stabilized file for writing"); while(1); }
  String header = "Date,Time,PWM,Voltage (V),Current (A),Power (W)";
  for (int i = 0; i < NUM_THERMISTORS; i++) header += "," + String(THERMISTOR_LABELS[i]);
  header += ",BME_Temp,BME_Hum,BME_Press";
  stabilizedDataFile.println(header);
  stabilizedDataFile.flush();
}

void setupDynamicFile() {
  DateTime now = rtc.now();
  sprintf(dynamicFileName, "/Dynamic_%04d%02d%02d_%02d%02d.csv", now.year(), now.month(), now.day(), now.hour(), now.minute());
  dynamicDataFile = SD.open(dynamicFileName, FILE_WRITE);
  if (!dynamicDataFile) { Serial.println("Failed to open dynamic file for writing"); while(1); }
  String header = "Date,Time,PWM,Voltage (V),Current (A),Power (W)";
  for (int i = 0; i < NUM_THERMISTORS; i++) header += "," + String(THERMISTOR_LABELS[i]);
  header += ",BME_Temp,BME_Hum,BME_Press";
  dynamicDataFile.println(header);
  dynamicDataFile.flush();
}

void writeLogEntry(const String& logEntry, File& file) {
  dataBuffer[bufferIndex++] = logEntry;
  if (bufferIndex >= BUFFER_SIZE) {
    for (int i = 0; i < BUFFER_SIZE; i++) file.println(dataBuffer[i]);
    file.flush();
    bufferIndex = 0;
  }
}

float applyEMA(float newReading, float previousFilteredValue, float alpha) {
  return (alpha * newReading) + ((1.0 - alpha) * previousFilteredValue);
}

/**
 * @brief Reads the ADC value from a thermistor pin with a sample averaging technique to reduce noise.
 * @param pin The analog pin to read from.
 * @return The averaged raw ADC reading.
 */
float readThermistorADC_Averaged(int pin) {
  const int numSamples = 10;
  float sum = 0;
  for (int i = 0; i < numSamples; i++) {
    sum += analogRead(pin);
  }
  return sum / numSamples;
}


void logData() {
  float voltage = ina260.readBusVoltage();
  float current = ina260.readCurrent();
  float power = ina260.readPower();
  float thermistorTemps[NUM_THERMISTORS];

  // Read and filter each thermistor's temperature
  for (int i = 0; i < NUM_THERMISTORS; i++) {
    float rawReading = readThermistorADC_Averaged(TEMP_SENSOR_PINS[i]);
    if (initialReadingsTaken) {
      filteredThermistorVals[i] = applyEMA(rawReading, filteredThermistorVals[i], EMA_ALPHA);
    } else {
      filteredThermistorVals[i] = rawReading;
    }

    // Check for extreme values after filtering
    if (filteredThermistorVals[i] <= 0 || filteredThermistorVals[i] >= ADC_MAX_VALUE) {
      thermistorTemps[i] = -999.0;
    } else {
      float R = SERIES_RESISTOR / ((float)ADC_MAX_VALUE / filteredThermistorVals[i] - 1.0);
      float steinhart = log(R / THERMISTOR_NOMINAL) / B_COEFFICIENT + 1.0 / (TEMPERATURE_NOMINAL + 273.15);
      thermistorTemps[i] = 1.0 / steinhart - 273.15;
    }
  }

  // Set the flag after the first full loop
  if (!initialReadingsTaken) {
    initialReadingsTaken = true;
  }

  float bmeTemp = bme.readTemperature();
  float bmeHum = bme.readHumidity();
  float bmePress = bme.readPressure() / 100.0F;
  DateTime now = rtc.now();

  String logEntry = String(now.year()) + "-" + String(now.month()) + "-" + String(now.day()) + "," +
                    String(now.hour()) + ":" + String(now.minute()) + ":" + String(now.second()) + ",";

  if (currentState == STATE_STABILIZED_SWEEP) {
    logEntry += String(FULL_PWM_SWEEP[currentPWMIndex]) + ",";
  } else if (currentState == STATE_DYNAMIC_SWEEP) {
    logEntry += String(DYNAMIC_PWM_SWEEP[currentPWMIndex]) + ",";
  } else {
    logEntry += "0,"; // During pre-calibration
  }

  logEntry += String(voltage / 1000.0, 3) + "," + String(current / 1000.0, 3) + "," + String(power / 1000.0, 3);
  for (int i = 0; i < NUM_THERMISTORS; i++) logEntry += "," + String(thermistorTemps[i], 2);
  logEntry += "," + String(bmeTemp, 2) + "," + String(bmeHum, 2) + "," + String(bmePress, 2);

  if (currentState == STATE_STABILIZED_SWEEP) {
    writeLogEntry(logEntry, stabilizedDataFile);
  } else if (currentState == STATE_DYNAMIC_SWEEP) {
    writeLogEntry(logEntry, dynamicDataFile);
  } else {
    // During pre-calibration, log to serial for debugging
    Serial.println(logEntry);
  }
}

void updateDisplay() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  DateTime now = rtc.now();
  display.printf("%04d-%02d-%02d %02d:%02d:%02d\n", now.year(), now.month(), now.day(), now.hour(), now.minute(), now.second());

  display.print("Status: ");
  switch (currentState) {
    case STATE_PRE_CALIBRATION:
      display.println("CALIBRATING");
      display.printf("Wait: %ds\n", (CALIBRATION_STABILITY_WINDOW_MS - (millis() - stableStartTime)) / 1000);
      break;
    case STATE_STABILIZED_SWEEP:
      display.print(isStable ? "STABLE" : "HEATING");
      display.printf(" PWM:%d\n", FULL_PWM_SWEEP[currentPWMIndex]);
      display.printf("Wait: %ds\n", (PROTOCOL_STABILITY_WINDOW_MS - (millis() - lastStabilityCheckTime)) / 1000);
      break;
    case STATE_DYNAMIC_SWEEP:
      display.println("DYNAMIC SWEEP");
      display.printf("PWM: %d\n", DYNAMIC_PWM_SWEEP[currentPWMIndex]);
      display.printf("Hold: %ds\n", (dynamicHoldTime - (millis() - dynamicStepStartTime)) / 1000);
      break;
    case STATE_COMPLETE:
      display.println("COMPLETE");
      break;
  }

  display.printf("Last dT: %.2fC\n", lastTempDelta);

  float bmeTemp = bme.readTemperature();
  float outsideTemp = -999.0; // Assume the thermistor is working
  if (initialReadingsTaken) {
    outsideTemp = readThermistorTemp(TEMP_SENSOR_PINS[6]);
  }
  display.printf("BME:%.2fC | Out:%.2fC\n", bmeTemp, outsideTemp);

  float power = ina260.readPower();
  display.printf("Power: %.2f W\n", power / 1000.0);
  display.display();
}

void setPWM(int duty) {
  ledc_set_duty(LEDC_HIGH_SPEED_MODE, (ledc_channel_t)HEATER_PWM_CHANNEL, duty);
  ledc_update_duty(LEDC_HIGH_SPEED_MODE, (ledc_channel_t)HEATER_PWM_CHANNEL);
}

void setupLEDCPWM() {
  ledc_timer_config_t ledc_timer = {
    .speed_mode = LEDC_HIGH_SPEED_MODE,
    .duty_resolution = (ledc_timer_bit_t)HEATER_PWM_RESOLUTION,
    .timer_num = (ledc_timer_t)HEATER_PWM_TIMER,
    .freq_hz = PWM_FREQ,
    .clk_cfg = LEDC_AUTO_CLK
  };
  ledc_timer_config(&ledc_timer);

  ledc_channel_config_t ledc_channel = {
    .gpio_num = HEATER_PWM_PIN,
    .speed_mode = LEDC_HIGH_SPEED_MODE,
    .channel = (ledc_channel_t)HEATER_PWM_CHANNEL,
    .intr_type = LEDC_INTR_DISABLE,
    .timer_sel = (ledc_timer_t)HEATER_PWM_TIMER,
    .duty = 0,
    .hpoint = 0
  };
  ledc_channel_config(&ledc_channel);
}

float readThermistorTemp(int pin) {
  // Find the filtered value for this thermistor
  int thermistorIndex = -1;
  for (int i = 0; i < NUM_THERMISTORS; i++) {
    if (TEMP_SENSOR_PINS[i] == pin) {
      thermistorIndex = i;
      break;
    }
  }

  if (thermistorIndex == -1) return -999.0;

  float raw = filteredThermistorVals[thermistorIndex];

  if (raw <= 0 || raw >= ADC_MAX_VALUE) return -999.0;

  float R = SERIES_RESISTOR / ((float)ADC_MAX_VALUE / raw - 1.0);
  float steinhart = log(R / THERMISTOR_NOMINAL) / B_COEFFICIENT + 1.0 / (TEMPERATURE_NOMINAL + 273.15);
  steinhart = 1.0 / steinhart - 273.15;
  return steinhart;
}

void performPreCalibration() {
  Serial.println(F("Starting pre-calibration..."));
  setPWM(0);

  // Initial fill of filtered values
  for (int i = 0; i < NUM_THERMISTORS; i++) {
    filteredThermistorVals[i] = readThermistorADC_Averaged(TEMP_SENSOR_PINS[i]);
  }

  float lastTemp = bme.readTemperature();
  calibrationStartTime = millis();
  stableStartTime = millis();

  // Set the correct state for the display
  currentState = STATE_PRE_CALIBRATION;

  while (true) {
    unsigned long currentTime = millis();
    if (currentTime - lastStabilityCheckTime >= DATA_UPDATE_INTERVAL_MS) {
      lastStabilityCheckTime = currentTime;
      float currentTemp = bme.readTemperature();
      float tempDelta = abs(currentTemp - lastTemp);
      lastTempDelta = tempDelta;
      logData(); // Log data even during calibration

      if (tempDelta < STABILITY_THRESHOLD_C) {
        if (stableStartTime == 0) stableStartTime = currentTime;
        if (currentTime - stableStartTime >= CALIBRATION_STABILITY_WINDOW_MS) break;
      } else {
        stableStartTime = 0;
      }
      lastTemp = currentTemp;
    }
    if (currentTime - calibrationStartTime >= MAX_CALIBRATION_TIMEOUT_MS) break;
    updateDisplay();
    delay(10);
  }

  Serial.println(F("Pre-calibration complete. System is at a stable ambient temperature."));
}
