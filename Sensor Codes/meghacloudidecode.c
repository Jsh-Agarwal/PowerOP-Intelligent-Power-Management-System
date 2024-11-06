#include <PZEM004Tv30.h>
#include <SPI.h>
#include <LiquidCrystal_I2C.h>

#ifdef ESP8266
#include <ESP8266WiFi.h>
#else
#include <WiFi.h>
#endif
#include <ModbusIP_ESP8266.h>

LiquidCrystal_I2C lcd(0x27, 20, 4); // set the LCD address to 0x27 for a 16 chars and 2 line display

/* Use software serial for the PZEM
   Pin 14 Rx D5(Connects to the Tx pin on the PZEM)
   Pin 12 Tx D6(Connects to the Rx pin on the PZEM)
*/
PZEM004Tv30 pzem(14, 12);

//ModbusIP object
ModbusIP mb;

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
  lcd.init();                      // initialize the lcd
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
  mb.addHreg(10);  //for power integer value
  mb.addHreg(11);  //for power decimal value
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

  //Update LCD
  lcd.setCursor(0, 1);
  lcd.print("V,A:" + String(voltage) + "V, " + String(current) + "A");
  lcd.setCursor(0, 2);
  lcd.print("P:" + String(power) + "W, E:" + String(energy) + "kWh");
  lcd.setCursor(0, 3);
  lcd.print("Fr:" + String(frequency) + "Hz, PF:" + String(pf));

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
