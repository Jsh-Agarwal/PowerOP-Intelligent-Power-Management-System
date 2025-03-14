from machine import UART, Pin, SoftI2C, ADC
import time, struct, dht, ssd1306

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
    #oled.text('Hum: {:.1f}%'.format(hum), 0, 10)
    oled.text('Volt: {:.1f}V'.format(voltage), 0, 10)
    oled.text('Curr: {:.3f}A'.format(current), 0, 20)
    oled.text('Power: {:.1f}W'.format(power), 0, 30)
    oled.text('E: {:.3f}kWh'.format(energy), 0, 40)
    oled.text('CO2: {}%'.format(analog_percent), 0, 50)
    oled.show()
    
    # Print to terminal
    
    print('Temp: {:.1f}C'.format(temp))
    print('Hum: {:.1f}%'.format(hum))
    print('Volt: {:.1f}V'.format(voltage))
    print('Curr: {:.3f}A'.format(current))
    print('Power: {:.1f}W'.format(power))
    print('E: {:.3f}kWh'.format(energy))
    print('Freq: {:.1f}Hz'.format(freq))
    print('PF: {:.1f}'.format(pf))
    print('Analog: {}%'.format(analog_percent))
    print('-' * 30)

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
    
    display_data(temp, hum, voltage, current, power, energy, freq, pf, analog_percent)
    time.sleep(2)
