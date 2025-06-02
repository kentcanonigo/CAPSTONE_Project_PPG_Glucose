#include <Arduino.h>

// --- Pin Configuration ---
// For ESP32, use valid ADC1 pins (e.g., GPIO32-39). Ensure they are distinct.
// Avoid ADC2 pins if using Wi-Fi.

// <<<<====== IMPORTANT: CHANGE THESE PINS TO YOUR ACTUAL ESP32 WIRING ======>>>>
const int PULSE_SENSOR_PIN_1 = 34; // Example: GPIO34 for Sensor 1 (ADC1_CH6)
const int PULSE_SENSOR_PIN_2 = 35; // Example: GPIO35 for Sensor 2 (ADC1_CH7)
const int PULSE_SENSOR_PIN_3 = 32; // Example: GPIO32 for Sensor 3 (ADC1_CH4)
// Common ADC1 pins for ESP32: 32, 33, 34, 35, 36 (VP), 39 (VN)
// Consult your specific ESP32 board's pinout diagram.

// --- Data Collection Variables ---
int collection_duration_s = 0;
unsigned long collection_start_time_ms = 0;
bool collecting = false;

// Target effective sampling rate (for each sensor, sequentially read)
const int TARGET_SAMPLING_RATE_HZ = 100;
const int SAMPLING_INTERVAL_MS = int(1000 / TARGET_SAMPLING_RATE_HZ);

// --- Setup Function ---
void setup()
{
    Serial.begin(115200);
    Serial.println("Analog Pulse Sensor Raw Data Collector (3 Sensors - ESP32)");

    pinMode(PULSE_SENSOR_PIN_1, INPUT);
    pinMode(PULSE_SENSOR_PIN_2, INPUT);
    pinMode(PULSE_SENSOR_PIN_3, INPUT);
    Serial.println("Sensor pins initialized.");
    Serial.println("Waiting for 'S,duration' command from host application...");
    Serial.println("Data format: Timestamp(ms),PPG_Finger1,PPG_Finger2,PPG_Finger3");
}

// --- Loop Function ---
void loop()
{
    if (Serial.available() > 0)
    {
        String command = Serial.readStringUntil('\n');
        command.trim();

        if (command.startsWith("S,"))
        {
            int commaIndex = command.indexOf(',');
            if (commaIndex != -1)
            {
                String durationStr = command.substring(commaIndex + 1);
                collection_duration_s = durationStr.toInt();
                if (collection_duration_s > 0)
                {
                    collection_start_time_ms = millis();
                    collecting = true;
                    Serial.println("ACK_START");
                }
                else
                {
                    Serial.println("ERR_DURATION");
                }
            }
            else
            {
                Serial.println("ERR_FORMAT_S");
            }
        }
    }

    if (collecting)
    {
        if (millis() - collection_start_time_ms < (unsigned long)collection_duration_s * 1000)
        {
            unsigned long current_reading_time_ms = millis();

            int ppgSignalValue1 = analogRead(PULSE_SENSOR_PIN_1);
            int ppgSignalValue2 = analogRead(PULSE_SENSOR_PIN_2);
            int ppgSignalValue3 = analogRead(PULSE_SENSOR_PIN_3);

            Serial.print(current_reading_time_ms);
            Serial.print(",");
            Serial.print(ppgSignalValue1);
            Serial.print(",");
            Serial.print(ppgSignalValue2);
            Serial.print(",");
            Serial.println(ppgSignalValue3);

            unsigned long processing_time_ms = millis() - current_reading_time_ms;
            long delay_needed_ms = SAMPLING_INTERVAL_MS - processing_time_ms;
            if (delay_needed_ms > 0)
            {
                delay(delay_needed_ms);
            }
        }
        else
        {
            collecting = false;
            Serial.println("ACK_DONE");
            collection_duration_s = 0;
        }
    }
}