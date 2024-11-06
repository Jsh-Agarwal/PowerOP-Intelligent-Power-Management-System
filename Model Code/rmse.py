import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# Load the data from the CSV file
data = pd.read_csv('sensor_data.csv', encoding='latin1')

# Convert the 'Day' column to datetime
data['Day'] = pd.to_datetime(data['Day'])

# Extract hourly and daily data
hourly_data = data.copy()
daily_data = hourly_data.resample('D', on='Day').mean().reset_index()

# Split the data into training, validation, and testing sets
train_size = 0.7
val_size = 0.15
test_size = 0.15

train_data, other_data = train_test_split(hourly_data, train_size=train_size, random_state=42)
val_data, test_data = train_test_split(other_data, test_size=test_size/(test_size+val_size), random_state=42)

# Define the features and target variable
X_train = train_data[['Current (A)', 'Inside Temperature (°C)']]
y_train = train_data['Power (W)']

X_val = val_data[['Current (A)', 'Inside Temperature (°C)']]
y_val = val_data['Power (W)']

X_test = test_data[['Current (A)', 'Inside Temperature (°C)']]
y_test = test_data['Power (W)']

# Create a dictionary to store model performance
models_rmse = {}

# Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
rf_y_val_pred = rf_regressor.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_y_val_pred))
models_rmse['Random Forest'] = rf_rmse

# XGBoost Regressor
xgb_regressor = xgb.XGBRegressor(random_state=42)
xgb_regressor.fit(X_train, y_train)
xgb_y_val_pred = xgb_regressor.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_y_val_pred))
models_rmse['XGBoost'] = xgb_rmse

# Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
linear_y_val_pred = linear_regressor.predict(X_val)
linear_rmse = np.sqrt(mean_squared_error(y_val, linear_y_val_pred))
models_rmse['Linear Regression'] = linear_rmse

# Robust Linear Regression
robust_regressor = HuberRegressor()
robust_regressor.fit(X_train, y_train)
robust_y_val_pred = robust_regressor.predict(X_val)
robust_rmse = np.sqrt(mean_squared_error(y_val, robust_y_val_pred))
models_rmse['Robust Linear Regression'] = robust_rmse

# Step-wise Linear Regression
stepwise_regressor = LinearRegression()
selector = RFE(stepwise_regressor, n_features_to_select=1, step=1)
selector.fit(X_train, y_train)
stepwise_y_val_pred = selector.predict(X_val)
stepwise_rmse = np.sqrt(mean_squared_error(y_val, stepwise_y_val_pred))
models_rmse['Step-wise Linear Regression'] = stepwise_rmse

# Plotting the RMSE values for different models
plt.figure(figsize=(10, 6))
plt.bar(models_rmse.keys(), models_rmse.values(), color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('RMSE Comparison of Different Models')
plt.ylim(0, max(models_rmse.values()) * 1.2)
plt.xticks(rotation=45)
plt.show()

# Displaying the RMSE values in a table
print('RMSE Values for Different Models:')
print(pd.DataFrame(models_rmse.items(), columns=['Model', 'RMSE']))

# Selecting the best model based on RMSE
best_model = min(models_rmse, key=models_rmse.get)
print(f'\nBest Model based on RMSE: {best_model}')
