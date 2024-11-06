import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# Load the data from the CSV file
data = pd.read_csv('sensor_data.csv', encoding='latin1')

# Convert the 'Day' column to datetime
data['Day'] = pd.to_datetime(data['Day'])

# Extract hourly and daily data
hourly_data = data.copy()
daily_data = hourly_data.resample('D', on='Day').mean().reset_index()

# Normalize the data
scaler = MinMaxScaler()
hourly_data_normalized = scaler.fit_transform(hourly_data.drop(['Day'], axis=1))
daily_data_normalized = scaler.fit_transform(daily_data.drop(['Day'], axis=1))

# Feature Engineering
hourly_data['Temperature Difference'] = hourly_data['Outside Temperature (°C)'] - hourly_data['Inside Temperature (°C)']
hourly_data['Humidity Difference'] = hourly_data['Outside Humidity (%)'] - hourly_data['Inside Humidity (%)']
# Interaction feature
hourly_data['Temp*Humidity Difference'] = hourly_data['Temperature Difference'] * hourly_data['Humidity Difference']
hourly_data['Power Factor * Current'] = hourly_data['Power Factor'] * hourly_data['Current (A)']

# Polynomial feature
hourly_data['Current Squared'] = hourly_data['Current (A)']**2

# Now include these new features in your model
# Split the data into train, validation, and test sets
X = hourly_data[['Current (A)', 'Inside Temperature (°C)', 'Outside Temperature (°C)', 'Inside Humidity (%)', 'Temperature Difference', 'Humidity Difference', 'Temp*Humidity Difference', 'Current Squared', 'Power Factor * Current']]
y = hourly_data['Power (W)']
# First split the data into a training set and a temporary set using an 80:20 ratio
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Then split the temporary set into validation and test sets using a 50:50 ratio
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the models you want to compare
models = {
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'KNN': KNeighborsRegressor(),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_mape = mean_absolute_percentage_error(y_val, y_val_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    
    print(f'{name} Validation Metrics:')
    print(f'MSE: {val_mse}, RMSE: {val_rmse}, MAE: {val_mae}, MAPE: {val_mape}')
    print(f'{name} Test Metrics:')
    print(f'MSE: {test_mse}, RMSE: {test_rmse}, MAE: {test_mae}, MAPE: {test_mape}')
    print('-' * 30)
    
    import matplotlib.pyplot as plt
    
    # Store the RMSE values in a dictionary
    rmse_values = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = np.sqrt(val_mse)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        
        # Store the RMSE values
        rmse_values[name] = test_rmse
        
        print(f'{name} Validation Metrics:')
        print(f'MSE: {val_mse}, RMSE: {val_rmse}')
        print(f'{name} Test Metrics:')
        print(f'MSE: {test_mse}, RMSE: {test_rmse}')
        print('-' * 30)
    
    # Plot the RMSE values
    plt.bar(rmse_values.keys(), rmse_values.values())
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title('RMSE of Different Models')
    plt.show()