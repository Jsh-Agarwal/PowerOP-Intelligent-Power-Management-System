from machine import UART, Pin, SoftI2C, ADC
import time, struct, dht, ssd1306, urequests as requests, json, network

# WiFi Credentials
WIFI_SSID = "Aditya"
WIFI_PASSWORD = "12345678"

# Supabase API details
SUPABASE_URL = "https://bwymfwjhgiorhcsgqpou.supabase.co/rest/v1/sensor"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ3eW1md2poZ2lvcmhjc2dxcG91Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzk2MjI5NTMsImV4cCI6MjA1NTE5ODk1M30.entM3-lgoZez8Yz9JOtsib3tWAd2EgnrYHX8TPgbjLo"

HEADERS = {
    "Content-Type": "application/json",
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Prefer": "return=minimal"
}

def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("Connecting to Wi-Fi...")
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        while not wlan.isconnected():
            pass
    print("Connected to Wi-Fi:", wlan.ifconfig())

connect_wifi()

# Initialize UART for PZEM-004T V3 (TX=GPIO13, RX=GPIO14)
uart = UART(2, 9600)

# Initialize DHT11 sensor
sensor = dht.DHT11(Pin(14))

# Initialize OLED display
i2c = SoftI2C(scl=Pin(22), sda=Pin(21))
oled = ssd1306.SSD1306_I2C(128, 64, i2c)

# Initialize ADC on GPIO34 (Analog Input)
adc = ADC(Pin(34))
adc.atten(ADC.ATTN_11DB)  # Set attenuation for full range (0-3.3V)

def read_measures():
    try:
        uart.write(b'\xF8\x04\x00\x00\x00\x0A\x64\x64') # Request data from PZEM
        time.sleep(0.1)
        payload = uart.read()
        if payload:
            payload = payload[3:-2]
            response_quantity = len(payload) // 2
            fmt = '>' + ('h' * response_quantity)
            return struct.unpack(fmt, payload)
    except:
        return None

def display_data(temp, hum, voltage, current, power, energy, freq, pf, analog_percent):
    oled.fill(0)  # Clear display
    oled.text('Temp: {:.1f}C'.format(temp), 0, 0)
    oled.text('Volt: {:.1f}V'.format(voltage), 0, 10)
    oled.text('Curr: {:.3f}A'.format(current), 0, 20)
    oled.text('Power: {:.1f}W'.format(power), 0, 30)
    oled.text('E: {:.3f}kWh'.format(energy), 0, 40)
    oled.text('CO2: {}%'.format(analog_percent), 0, 50)
    oled.show()
    
    # Print to terminal
    print('Temp: {:.1f}C'.format(temp))
    print('Volt: {:.1f}V'.format(voltage))
    print('Curr: {:.3f}A'.format(current))
    print('Power: {:.1f}W'.format(power))
    print('E: {:.3f}kWh'.format(energy))
    print('Freq: {:.1f}Hz'.format(freq))
    print('PF: {:.1f}'.format(pf))
    print('Analog: {}%'.format(analog_percent))
    print('-' * 30)

def send_to_supabase(data):
    # Ensure correct data types for numeric fields
    formatted_data = {
        "temperature": float(data["temperature"]),
        "humidity": float(data["humidity"]),
        "voltage": float(data["voltage"]),
        "current": float(data["current"]),
        "power": float(data["power"]),
        "energy": float(data["energy"]),
        "frequency": float(data["frequency"]),
        "power_factor": float(data["power_factor"]),
        "co2": int(data["co2"])  # CO2 is likely a percentage, keeping it as an integer
    }

    json_payload = json.dumps(formatted_data)  # Send as a single object
    try:
        print(f"Sending JSON: {json_payload}")
        response = requests.post(SUPABASE_URL, data=json_payload, headers=HEADERS)
        print(f"Response: {response.status_code} - {response.text}")
        if response.status_code == 201:
            print("[SUCCESS] Data sent")
        else:
            print(f"[ERROR] {response.status_code}: {response.text}")
        response.close()
    except Exception as e:
        print(f"[ERROR] {e}")

while True:
    try:
        # Read temperature and humidity
        sensor.measure()
        temp = sensor.temperature()
        hum = sensor.humidity()
    except OSError:
        temp, hum = 0, 0

    try:
        # Read power meter data
        all_measures = read_measures()
        if all_measures:
            voltage = all_measures[0] / 10.0
            current = ((all_measures[2] << 16) | all_measures[1]) / 1000.0
            power = ((all_measures[4] << 16) | all_measures[3]) / 10.0
            energy = ((all_measures[6] << 16) | all_measures[5]) / 1000.0
            freq = all_measures[7] / 10.0
            pf = all_measures[8] / 10.0
        else:
            voltage, current, power, energy, freq, pf = 0, 0, 0, 0, 0, 0
    except:
        voltage, current, power, energy, freq, pf = 0, 0, 0, 0, 0, 0

    # Read analog value from GPIO34 and convert to percentage
    analog_value = adc.read()
    analog_percent = int((analog_value / 4095) * 100)
    
    # Prepare data for Supabase
    sensor_data = {
        "temperature": temp,
        "humidity": hum,
        "voltage": voltage,
        "current": current,
        "power": power,
        "energy": energy,
        "frequency": freq,
        "power_factor": pf,
        "co2": analog_percent
    }
    
    send_to_supabase(sensor_data)
    display_data(temp, hum, voltage, current, power, energy, freq, pf, analog_percent)
    time.sleep(5)
