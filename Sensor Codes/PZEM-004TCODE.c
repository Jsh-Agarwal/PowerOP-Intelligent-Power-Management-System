#include <PZEM004Tv30.h>

/* Define RX and TX pins for PZEM module */
#define PZEM_RX_PIN 16
#define PZEM_TX_PIN 17

/* Use hardware serial for the PZEM */
#define PZEM_SERIAL Serial2 // Assuming you're using Serial2 on ESP32

PZEM004Tv30 pzem(&PZEM_SERIAL, PZEM_RX_PIN, PZEM_TX_PIN);

void setup() {
    /* Debugging serial */
    // Serial.begin(115200);
    Serial.begin(9600);

    /* Initialize the PZEM serial communication */
    PZEM_SERIAL.begin(9600, SERIAL_8N1, PZEM_RX_PIN, PZEM_TX_PIN);
}

void loop() {

    Serial.print("Custom Address:");
    Serial.println(pzem.readAddress(), HEX);

    // Read the data from the sensor
    float voltage = pzem.voltage();
    float current = pzem.current();
    float power = pzem.power();
    float energy = pzem.energy();
    float frequency = pzem.frequency();
    float pf = pzem.pf();

    // Check if the data is valid
    if (isnan(voltage)) {
        Serial.println("Error reading voltage");
    } else if (isnan(current)) {
        Serial.println("Error reading current");
    } else if (isnan(power)) {
        Serial.println("Error reading power");
    } else if (isnan(energy)) {
        Serial.println("Error reading energy");
    } else if (isnan(frequency)) {
        Serial.println("Error reading frequency");
    } else if (isnan(pf)) {
        Serial.println("Error reading power factor");
    } else {

        // Print the values to the Serial console
        Serial.print("Voltage: ");      Serial.print(voltage);      Serial.println("V");
        Serial.print("Current: ");      Serial.print(current);      Serial.println("A");
        Serial.print("Power: ");        Serial.print(power);        Serial.println("W");
        Serial.print("Energy: ");       Serial.print(energy, 3);     Serial.println("kWh");
        Serial.print("Frequency: ");    Serial.print(frequency, 1); Serial.println("Hz");
        Serial.print("PF: ");           Serial.println(pf);
    }

    Serial.println();
    delay(2000);
}
