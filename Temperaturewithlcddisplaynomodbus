#include <PZEM004Tv30.h>
#include <SPI.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <DHT.h>

#ifdef ESP8266
#include <ESP8266WiFi.h>
#else
#include <WiFi.h>
#endif

#define DHTPIN 2     // Digital pin connected to the DHT sensor

// Uncomment the type of sensor in use:
#define DHTTYPE DHT11   // DHT 11

LiquidCrystal_I2C lcd(0x27, 20, 4); // set the LCD address to 0x27 for a 16 chars and 2 line display
DHT dht(DHTPIN, DHTTYPE);

/* Use software serial for the PZEM
   Pin 14 Rx D5(Connects to the Tx pin on the PZEM)
   Pin 12 Tx D6(Connects to the Rx pin on the PZEM)
*/
PZEM004Tv30 pzem(&Serial1, 14, 12);

void setup() {
  Serial.begin(9600);
  WiFi.begin("WIFI", "PASSWORD");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  //LCD
  lcd.begin();                      // initialize the lcd
  //Print a message to the LCD.
  lcd.backlight();
  lcd.setCursor(1, 0);
  lcd.print("Energy Monitoring");
  lcd.setCursor(0, 1);
  lcd.print("V,A:000.00V, 0.0A");
  lcd.setCursor(0, 2);
  lcd.print("P:0.00W, E:0.000kWh");
  lcd.setCursor(0, 3);
  lcd.print("Fr:00.00Hz, PF:0.00");

  // DHT Sensor
  dht.begin();
}

void loop() {
  float voltage = pzem.voltage();
  if (!isnan(voltage)) {
    Serial.print("Voltage: "); Serial.print(voltage); Serial.println("V");
  } else {
    Serial.println("Error reading voltage");
  }

  float current = pzem.current();
  if (!isnan(current)) {
    Serial.print("Current: "); Serial.print(current); Serial.println("A");
  } else {
    Serial.println("Error reading current");
  }

  float power = pzem.power();
  if (!isnan(power)) {
    Serial.print("Power: "); Serial.print(power); Serial.println("W");
  } else {
    Serial.println("Error reading power");
  }

  float energy = pzem.energy();
  if (!isnan(energy)) {
    Serial.print("Energy: "); Serial.print(energy, 3); Serial.println("kWh");
  } else {
    Serial.println("Error reading energy");
  }

  float frequency = pzem.frequency();
  if (!isnan(frequency)) {
    Serial.print("Frequency: "); Serial.print(frequency, 1); Serial.println("Hz");
  } else {
    Serial.println("Error reading frequency");
  }

  float pf = pzem.pf();
  if (!isnan(pf)) {
    Serial.print("PF: "); Serial.println(pf);
  } else {
    Serial.println("Error reading power factor");
  }

  // DHT Sensor
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  if (!isnan(humidity) && !isnan(temperature)) {
    Serial.print("Humidity: "); Serial.print(humidity); Serial.println("%");
    Serial.print("Temperature: "); Serial.print(temperature); Serial.println("°C");
  } else {
    Serial.println("Error reading DHT sensor data");
  }

  Serial.println();

  //Update LCD
  lcd.setCursor(0, 1);
  lcd.print("V,A:" + String(voltage) + "V, " + String(current) + "A");
  lcd.setCursor(0, 2);
  lcd.print("P:" + String(power) + "W, E:" + String(energy) + "kWh");
  lcd.setCursor(0, 3);
  lcd.print("Fr:" + String(frequency) + "Hz, PF:" + String(pf));

  // Display DHT data on LCD
  lcd.setCursor(0, 3);
  lcd.print("Temp:" + String(temperature) + "C, Hum:" + String(humidity) + "%");

  delay(2000);
}
