import pandas as pd

# Define data
data = {
    'current': [0.1, 0.2, 0.3, 0.4, 0.5],
    'temperature': [30, 35, 40, 45, 50],
    'power': [100, 200, 300, 400, 500]
}

# Create DataFrame
df = pd.DataFrame(data)

# Write DataFrame to CSV file
df.to_csv('sensor_data.csv', index=False)