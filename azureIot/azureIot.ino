#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <DHT.h>
#include <PZEM004Tv30.h>
#include <ArduinoJson.h>

// WiFi Credentials
const char *ssid = "Nothing phone (2a)";
const char *password = "JAGDISH3456";

// Azure IoT Hub Credentials (Using SAS Token)
#define IOT_HUB_NAME "SmartHVAC.azure-devices.net"
#define DEVICE_ID "ESP32"
#define SAS_TOKEN "SharedAccessSignature sr=SmartHVAC.azure-devices.net/devices/ESP32&sig=KylaioyIzkrmiFcDC9I9P/8yx%2BCVE5LNgeuSDmSBD8g%3D&se=1739527777"

// MQTT Broker settings
const char *mqttServer = IOT_HUB_NAME;
const int mqttPort = 8883;
WiFiClientSecure wifiClient;
PubSubClient mqttClient(wifiClient);

// DHT11 Sensor
#define DHTPIN 26
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// PZEM Sensor
PZEM004Tv30 pzem(&Serial1, 16, 17);

// Function to connect WiFi
void connectWiFi() {
  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
}

// Function to connect to Azure IoT Hub using SAS Token
void connectAzureIoTHub() {
  Serial.print("Connecting to Azure IoT Hub...");
  wifiClient.setInsecure(); // Ignore SSL verification (not recommended for production)
  mqttClient.setServer(mqttServer, mqttPort);

  while (!mqttClient.connected()) {
    String clientId = DEVICE_ID;
    String username = String(IOT_HUB_NAME) + "/" + DEVICE_ID + "/?api-version=2021-04-12";
    String password = SAS_TOKEN;

    if (mqttClient.connect(clientId.c_str(), username.c_str(), password.c_str())) {
      Serial.println("Connected to Azure IoT Hub using SAS Token!");
    } else {
      Serial.print("Failed to connect. Retrying in 5s...");
      delay(5000);
    }
  }
}

// Function to read sensor data
void readSensors(float &temperature, float &humidity, float &voltage, float &current, float &power) {
  temperature = dht.readTemperature();
  humidity = dht.readHumidity();
  voltage = pzem.voltage();
  current = pzem.current();
  power = pzem.power();
}

// Function to send data to Azure IoT Hub
void sendTelemetry() {
  float temperature, humidity, voltage, current, power;
  readSensors(temperature, humidity, voltage, current, power);

  StaticJsonDocument<256> doc;
  doc["temperature"] = temperature;
  doc["humidity"] = humidity;
  doc["voltage"] = voltage;
  doc["current"] = current;
  doc["power"] = power;

  char message[256];
  serializeJson(doc, message);

  if (mqttClient.publish("devices/ESP32/messages/events/", message)) {
    Serial.println("Telemetry sent to Azure IoT Hub!");
  } else {
    Serial.println("Failed to send telemetry!");
  }
}

// Setup function
void setup() {
  Serial.begin(9600);
  Serial.println("Hello, ESP32!");
  dht.begin();
  
  connectWiFi();
  connectAzureIoTHub();
}

// Loop function
void loop() {
  if (!mqttClient.connected()) {
    connectAzureIoTHub();
  }
  mqttClient.loop();
  
  sendTelemetry();
  delay(5000);
}
