import random
import csv
from datetime import datetime, timedelta

# Define the range and resolution for each sensor parameter
voltage_range = (220, 240)  # Mumbai voltage range
current_range = (5, 15)  # Assuming AC current range in amps
power_factor_range = (0.8, 1.0)  # Typical power factor for AC units
frequency_range = (49, 51)  # Mumbai frequency range
energy_range = (0, 9999.99)

# Define temperature and humidity ranges for different seasons
summer_temp_inside_range = (16, 27)  # Typical indoor temperature AC range in Mumbai (Summer)
summer_temp_outside_range = (32, 38)  # Typical outdoor temperature range in Mumbai (Summer)
summer_humidity_range = (60, 75)  # Typical humidity range in Mumbai (Summer)

monsoon_temp_inside_range = (22, 28)  # Typical indoor temperature AC range in Mumbai (Monsoon)
monsoon_temp_outside_range = (25, 32)  # Typical outdoor temperature range in Mumbai (Monsoon)
monsoon_humidity_range = (70, 90)  # Typical humidity range in Mumbai (Monsoon)

winter_temp_inside_range = (18, 24)  # Typical indoor temperature AC range in Mumbai (Winter)
winter_temp_outside_range = (20, 28)  # Typical outdoor temperature range in Mumbai (Winter)
winter_humidity_range = (40, 60)  # Typical humidity range in Mumbai (Winter)

# Resolution for each sensor parameter
resolution_voltage = 0.1
resolution_current = 0.01
resolution_power_factor = 0.01
resolution_frequency = 0.1
resolution_energy = 1

# Generate artificial sensor data for the AC unit
num_samples = 5000  # You can adjust the number of samples as needed
sensor_data = []
start_date = datetime(2023, 6, 1)  # Start date: June 1, 2023 (Summer)

# Initialize previous values
prev_voltage = None
prev_current = None
prev_power_factor = None
prev_frequency = None
prev_energy = None
prev_temperature_inside = None
prev_temperature_outside = None
prev_humidity_inside = None
prev_humidity_outside = None

for i in range(num_samples):
    # Generate data for electrical parameters
    if prev_voltage is None:
        voltage = round(random.uniform(*voltage_range), 1)
    else:
        voltage = round(prev_voltage + random.uniform(-0.5, 0.5), 1)
        voltage = max(voltage_range[0], min(voltage_range[1], voltage))

    if prev_current is None:
        current = round(random.uniform(*current_range), 2)
    else:
        current = round(prev_current + random.uniform(-0.5, 0.5), 2)
        current = max(current_range[0], min(current_range[1], current))

    if prev_power_factor is None:
        power_factor = round(random.uniform(*power_factor_range), 2)
    else:
        power_factor = round(prev_power_factor + random.uniform(-0.05, 0.05), 2)
        power_factor = max(power_factor_range[0], min(power_factor_range[1], power_factor))

    if prev_frequency is None:
        frequency = round(random.uniform(*frequency_range), 1)
    else:
        frequency = round(prev_frequency + random.uniform(-0.2, 0.2), 1)
        frequency = max(frequency_range[0], min(frequency_range[1], frequency))

    if prev_energy is None:
        energy = round(random.uniform(*energy_range), 0)
    else:
        energy = round(prev_energy + random.uniform(0, 100), 0)
        energy = min(energy_range[1], energy)

    # Generate data for temperature and humidity based on the season
    date = start_date + timedelta(days=i)
    month = date.month

    if month in (3, 4, 5):  # Summer
        if prev_temperature_inside is None:
            temperature_inside = round(random.uniform(*summer_temp_inside_range), 1)
        else:
            temperature_inside = round(prev_temperature_inside + random.uniform(-0.5, 0.5), 1)
            temperature_inside = max(summer_temp_inside_range[0], min(summer_temp_inside_range[1], temperature_inside))

        if prev_temperature_outside is None:
            temperature_outside = round(random.uniform(*summer_temp_outside_range), 1)
        else:
            temperature_outside = round(prev_temperature_outside + random.uniform(-0.5, 0.5), 1)
            temperature_outside = max(summer_temp_outside_range[0], min(summer_temp_outside_range[1], temperature_outside))

        if prev_humidity_inside is None:
            humidity_inside = round(random.uniform(*summer_humidity_range), 1)
        else:
            humidity_inside = round(prev_humidity_inside + random.uniform(-2, 2), 1)
            humidity_inside = max(summer_humidity_range[0], min(summer_humidity_range[1], humidity_inside))

        if prev_humidity_outside is None:
            humidity_outside = round(random.uniform(*summer_humidity_range), 1)
        else:
            humidity_outside = round(prev_humidity_outside + random.uniform(-2, 2), 1)
            humidity_outside = max(summer_humidity_range[0], min(summer_humidity_range[1], humidity_outside))

    elif month in (6, 7, 8, 9):  # Monsoon
        if prev_temperature_inside is None:
            temperature_inside = round(random.uniform(*monsoon_temp_inside_range), 1)
        else:
            temperature_inside = round(prev_temperature_inside + random.uniform(-0.5, 0.5), 1)
            temperature_inside = max(monsoon_temp_inside_range[0], min(monsoon_temp_inside_range[1], temperature_inside))

        if prev_temperature_outside is None:
            temperature_outside = round(random.uniform(*monsoon_temp_outside_range), 1)
        else:
            temperature_outside = round(prev_temperature_outside + random.uniform(-0.5, 0.5), 1)
            temperature_outside = max(monsoon_temp_outside_range[0], min(monsoon_temp_outside_range[1], temperature_outside))

        if prev_humidity_inside is None:
            humidity_inside = round(random.uniform(*monsoon_humidity_range), 1)
        else:
            humidity_inside = round(prev_humidity_inside + random.uniform(-2, 2), 1)
            humidity_inside = max(monsoon_humidity_range[0], min(monsoon_humidity_range[1], humidity_inside))

        if prev_humidity_outside is None:
            humidity_outside = round(random.uniform(*monsoon_humidity_range), 1)
        else:
            humidity_outside = round(prev_humidity_outside + random.uniform(-2, 2), 1)
            humidity_outside = max(monsoon_humidity_range[0], min(monsoon_humidity_range[1], humidity_outside))

    else:  # Winter
        if prev_temperature_inside is None:
            temperature_inside = round(random.uniform(*winter_temp_inside_range), 1)
        else:
            temperature_inside = round(prev_temperature_inside + random.uniform(-0.5, 0.5), 1)
            temperature_inside = max(winter_temp_inside_range[0], min(winter_temp_inside_range[1], temperature_inside))

        if prev_temperature_outside is None:
            temperature_outside = round(random.uniform(*winter_temp_outside_range), 1)
        else:
            temperature_outside = round(prev_temperature_outside + random.uniform(-0.5, 0.5), 1)
            temperature_outside = max(winter_temp_outside_range[0], min(winter_temp_outside_range[1], temperature_outside))

        if prev_humidity_inside is None:
            humidity_inside = round(random.uniform(*winter_humidity_range), 1)
        else:
            humidity_inside = round(prev_humidity_inside + random.uniform(-2, 2), 1)
            humidity_inside = max(winter_humidity_range[0], min(winter_humidity_range[1], humidity_inside))

        if prev_humidity_outside is None:
            humidity_outside = round(random.uniform(*winter_humidity_range), 1)
        else:
            humidity_outside = round(prev_humidity_outside + random.uniform(-2, 2), 1)
            humidity_outside = max(winter_humidity_range[0], min(winter_humidity_range[1], humidity_outside))

    # Calculate power based on current, voltage, and power factor
    power = round(voltage * current * power_factor, 0)

    # Get the current date and time
    day = date.strftime("%Y-%m-%d")
    hours = str(random.randint(0, 23)).zfill(2)  # Generate random hours (0-23)
    minutes = str(random.randint(0, 59)).zfill(2)  # Generate random minutes (0-59)

    sensor_data.append([day, hours, minutes, voltage, current, power, power_factor, frequency, energy, temperature_inside, temperature_outside, humidity_inside, humidity_outside])

    sensor_data.append([day, hours, minutes, voltage, current, power, power_factor, frequency, energy, temperature_inside, temperature_outside, humidity_inside, humidity_outside])

    # Update previous values
    prev_voltage = voltage
    prev_current = current
    prev_power_factor = power_factor
    prev_frequency = frequency
    prev_energy = energy
    prev_temperature_inside = temperature_inside
    prev_temperature_outside = temperature_outside
    prev_humidity_inside = humidity_inside
    prev_humidity_outside = humidity_outside

# Write the data to a CSV file
csv_filename = 'sensor_data.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Day', 'Hours', 'Minutes', 'Voltage (V)', 'Current (A)', 'Power (W)', 'Power Factor', 'Frequency (Hz)', 'Energy (kWh)', 'Inside Temperature (°C)', 'Outside Temperature (°C)', 'Inside Humidity (%)', 'Outside Humidity (%)'])
    csvwriter.writerows(sensor_data)

print(f"Sensor data for the AC unit in Mumbai generated and saved to {csv_filename}.")

# Print a sample of the generated data
sample_data = sensor_data[:10]  # Print the first 10 samples for demonstration
print("\nSample of Generated Sensor Data:")
for row in sample_data:
    print(row)

# import random
# import csv
# from datetime import datetime, timedelta
# import requests

# # Define the range and resolution for each sensor parameter
# voltage_range = (220, 240)  # Mumbai voltage range
# current_range = (5, 15)  # Assuming AC current range in amps
# power_factor_range = (0.8, 1.0)  # Typical power factor for AC units
# frequency_range = (49, 51)  # Mumbai frequency range

# # Define temperature and humidity ranges for different seasons
# summer_temp_inside_range = (24, 30)  # Typical indoor temperature AC range in Mumbai (Summer)
# monsoon_temp_inside_range = (22, 28)  # Typical indoor temperature AC range in Mumbai (Monsoon)
# winter_temp_inside_range = (18, 24)  # Typical indoor temperature AC range in Mumbai (Winter)

# # Resolution for each sensor parameter
# resolution_voltage = 0.1
# resolution_current = 0.01
# resolution_power_factor = 0.01
# resolution_frequency = 0.1

# # Generate artificial sensor data for the AC unit
# num_samples = 1000  # You can adjust the number of samples as needed
# sensor_data = []
# start_date = datetime(2023, 6, 1)  # Start date: June 1, 2023 (Summer)

# # Initialize previous values
# prev_voltage = None
# prev_current = None
# prev_power_factor = None
# prev_frequency = None
# prev_energy = 0

# # Get user input for the location
# location = input("Enter the location (city, country): ")

# # Initialize API key and URL for OpenWeatherMap
# api_key = "YOUR_API_KEY"  # Replace with your actual API key
# base_url = "http://api.openweathermap.org/data/2.5/weather?"

# for i in range(num_samples):
#     # Generate data for electrical parameters
#     if prev_voltage is None:
#         voltage = round(random.uniform(*voltage_range), 1)
#     else:
#         voltage = round(prev_voltage + random.uniform(-0.5, 0.5), 1)
#         voltage = max(voltage_range[0], min(voltage_range[1], voltage))

#     if prev_current is None:
#         current = round(random.uniform(*current_range), 2)
#     else:
#         current = round(prev_current + random.uniform(-0.5, 0.5), 2)
#         current = max(current_range[0], min(current_range[1], current))

#     if prev_power_factor is None:
#         power_factor = round(random.uniform(*power_factor_range), 2)
#     else:
#         power_factor = round(prev_power_factor + random.uniform(-0.05, 0.05), 2)
#         power_factor = max(power_factor_range[0], min(power_factor_range[1], power_factor))

#     if prev_frequency is None:
#         frequency = round(random.uniform(*frequency_range), 1)
#     else:
#         frequency = round(prev_frequency + random.uniform(-0.2, 0.2), 1)
#         frequency = max(frequency_range[0], min(frequency_range[1], frequency))

#     # Calculate power based on current, voltage, and power factor
#     power = round(voltage * current * power_factor, 0)

#     # Calculate energy based on power and a random factor
#     energy = prev_energy + power * random.uniform(0.5, 2.0)

#     # Get the current date and time
#     date = start_date + timedelta(days=i)
#     day = date.strftime("%Y-%m-%d")
#     hours = str((i * 24 // num_samples)).zfill(2)  # Generate sequential hours (0-23)
#     minutes = str((i * 1440 // num_samples) % 60).zfill(2)  # Generate sequential minutes (0-59)

#     # Generate data for temperature and humidity based on the season
#     month = date.month
#     if month in (3, 4, 5):  # Summer
#         temperature_inside = round(random.uniform(*summer_temp_inside_range), 1)
#     elif month in (6, 7, 8, 9):  # Monsoon
#         temperature_inside = round(random.uniform(*monsoon_temp_inside_range), 1)
#     else:  # Winter
#         temperature_inside = round(random.uniform(*winter_temp_inside_range), 1)

#     # Fetch outside temperature and humidity from OpenWeatherMap API
#     complete_url = f"{base_url}appid={api_key}&q={location}&units=metric"
#     response = requests.get(complete_url)
#     data = response.json()

#     if data["cod"] == 200:
#         temperature_outside = round(data["main"]["temp"], 1)
#         humidity_outside = round(data["main"]["humidity"], 1)
#     else:
#         print(f"Error fetching weather data: {data['message']}")
#         temperature_outside = None
#         humidity_outside = None

#     sensor_data.append([day, hours, minutes, voltage, current, power, power_factor, frequency, round(energy, 2), temperature_inside, temperature_outside, humidity_inside, humidity_outside])

#     # Update previous values
#     prev_voltage = voltage
#     prev_current = current
#     prev_power_factor = power_factor
#     prev_frequency = frequency
#     prev_energy = energy

# # Write the data to a CSV file
# csv_filename = 'sensor_data.csv'
# with open(csv_filename, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(['Day', 'Hours', 'Minutes', 'Voltage (V)', 'Current (A)', 'Power (W)', 'Power Factor', 'Frequency (Hz)', 'Energy (kWh)', 'Inside Temperature (°C)', 'Outside Temperature (°C)', 'Inside Humidity (%)', 'Outside Humidity (%)'])
#     csvwriter.writerows(sensor_data)

# print(f"Sensor data for the AC unit generated and saved to {csv_filename}.")

# # Print a sample of the generated data
# sample_data = sensor_data[:10]  # Print the first 10 samples for demonstration
# print("\nSample of Generated Sensor Data:")
# for row in sample_data:
#     print(row)