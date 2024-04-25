#include <PZEM004Tv30.h>
#include <SPI.h>
#include <Wire.h>
#include <ModbusIP_ESP8266.h>

#ifdef ESP8266
#include <ESP8266WiFi.h>
#else
#include <WiFi.h>
#endif

// Use hardware serial for the PZEM
HardwareSerial pzemSerial(1);

PZEM004Tv30 pzem(&pzemSerial, 16, 17); // RX, TX pins for PZEM004Tv30

ModbusIP mb;

void setup() {
  Serial.begin(9600);
  WiFi.begin("realme 7", "JAGDISH34");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  pzemSerial.begin(9600); // Initialize the hardware serial

  mb.server();
  //configure holding register for data
  mb.addHreg(0);  //for voltage integer value
  mb.addHreg(1);  //for voltage decimal value
  mb.addHreg(2);  //for current integer value
  mb.addHreg(3);  //for current decimal value
  mb.addHreg(4);  //for power integer value
  mb.addHreg(5);  //for power decimal value
  mb.addHreg(6);  //for energy integer value
  mb.addHreg(7);  //for energy decimal value
  mb.addHreg(8);  //for frequency integer value
  mb.addHreg(9);  //for frequency decimal value
  mb.addHreg(10); //for power integer value
  mb.addHreg(11); //for power decimal value
}

void loop() {
  float voltage = pzem.voltage();
  if ( !isnan(voltage) ) {
    Serial.print("Voltage: "); Serial.print(voltage); Serial.println("V");
  } else {
    Serial.println("Error reading voltage");
  }

  float current = pzem.current();
  if ( !isnan(current) ) {
    Serial.print("Current: "); Serial.print(current); Serial.println("A");
  } else {
    Serial.println("Error reading current");
  }

  float power = pzem.power();
  if ( !isnan(power) ) {
    Serial.print("Power: "); Serial.print(power); Serial.println("W");
  } else {
    Serial.println("Error reading power");
  }

  float energy = pzem.energy();
  if ( !isnan(energy) ) {
    Serial.print("Energy: "); Serial.print(energy, 3); Serial.println("kWh");
  } else {
    Serial.println("Error reading energy");
  }

  float frequency = pzem.frequency();
  if ( !isnan(frequency) ) {
    Serial.print("Frequency: "); Serial.print(frequency, 1); Serial.println("Hz");
  } else {
    Serial.println("Error reading frequency");
  }

  float pf = pzem.pf();
  if ( !isnan(pf) ) {
    Serial.print("PF: "); Serial.println(pf);
  } else {
    Serial.println("Error reading power factor");
  }

  Serial.println();

  //Write to Holding register
  //voltage
  int v_int = int(voltage);
  int v_dec = int((voltage - v_int) * 100);
  mb.Hreg(0, v_int);
  mb.Hreg(1, v_dec);
  delay(50);
  //current
  int c_int = int(current);
  int c_dec = int((current - c_int) * 100);
  mb.Hreg(2, c_int);
  mb.Hreg(3, c_dec);
  delay(50);
  //power
  int p_int = int(power);
  int p_dec = int((power - p_int) * 100);
  mb.Hreg(4, p_int);
  mb.Hreg(5, p_dec);
  delay(50);
  //energy
  int e_int = int(energy);
  int e_dec = int((energy - e_int) * 1000);
  mb.Hreg(6, e_int);
  mb.Hreg(7, e_dec);
  delay(50);
  //frequency
  int f_int = int(frequency);
  int f_dec = int((frequency - f_int) * 10);
  mb.Hreg(8, f_int);
  mb.Hreg(9, f_dec);
  delay(50);
  //pf
  int pf_int = int(pf);
  int pf_dec = int((pf - pf_int) * 100);
  mb.Hreg(10, pf_int);
  mb.Hreg(11, pf_dec);    
  //Call once inside loop()
  mb.task();

  delay(2000);
}
