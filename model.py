# # # import pandas as pd
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.ensemble import RandomForestRegressor
# # # from sklearn.metrics import mean_squared_error
# # # from xgboost import XGBRegressor
# # # from sklearn.linear_model import LinearRegression, RANSACRegressor
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from sklearn.preprocessing import MinMaxScaler

# # # # Load the data from the CSV file
# # # data = pd.read_csv('sensor_data.csv', encoding='latin1')

# # # # Convert the 'Day' column to datetime
# # # data['Day'] = pd.to_datetime(data['Day'])

# # # # Extract hourly and daily data
# # # hourly_data = data.copy()
# # # daily_data = hourly_data.resample('D', on='Day').mean().reset_index()

# # # # Normalize the data
# # # scaler = MinMaxScaler()
# # # hourly_data_normalized = scaler.fit_transform(hourly_data.drop(['Day'], axis=1))
# # # daily_data_normalized = scaler.fit_transform(daily_data.drop(['Day'], axis=1))

# # # # Plot all variables in one graph for hourly data (random)
# # # plt.figure(figsize=(12, 8))
# # # plt.plot(hourly_data.drop(['Day'], axis=1))
# # # plt.legend(hourly_data.columns)
# # # plt.title('Hourly Random Variables')
# # # plt.xlabel('Time')
# # # plt.ylabel('Value')
# # # plt.xticks(rotation=45)
# # # plt.show()

# # # # Plot all variables in one graph for daily data (random)
# # # plt.figure(figsize=(12, 8))
# # # plt.plot(daily_data.drop(['Day'], axis=1))
# # # plt.legend(daily_data.columns)
# # # plt.title('Daily Random Variables')
# # # plt.xlabel('Date')
# # # plt.ylabel('Value')
# # # plt.xticks(rotation=45)
# # # plt.show()

# # # # Plot all variables in one graph for hourly data (normalized)
# # # plt.figure(figsize=(12, 8))
# # # plt.plot(hourly_data_normalized)
# # # plt.legend(hourly_data.columns)
# # # plt.title('Hourly Normalized Variables')
# # # plt.xlabel('Time')
# # # plt.ylabel('Normalized Value')
# # # plt.xticks(rotation=45)
# # # plt.show()

# # # # Plot all variables in one graph for daily data (normalized)
# # # plt.figure(figsize=(12, 8))
# # # plt.plot(daily_data_normalized)
# # # plt.legend(daily_data.columns)
# # # plt.title('Daily Normalized Variables')
# # # plt.xlabel('Date')
# # # plt.ylabel('Normalized Value')
# # # plt.xticks(rotation=45)
# # # plt.show()

# # # # Plot temperature inside versus power using a hexbin plot
# # # plt.figure(figsize=(10, 6))
# # # plt.hexbin(hourly_data['Inside Temperature (°C)'], hourly_data['Power (W)'], gridsize=15, cmap='inferno')
# # # plt.colorbar()
# # # plt.title('Temperature Inside vs. Power')
# # # plt.xlabel('Temperature Inside (°C)')
# # # plt.ylabel('Power (W)')
# # # plt.show()

# # # # Split the data into training, validation, and testing sets
# # # train_size = 0.7
# # # val_size = 0.15
# # # test_size = 0.15

# # # train_data, other_data = train_test_split(hourly_data, train_size=train_size, random_state=42)
# # # val_data, test_data = train_test_split(other_data, test_size=test_size/(test_size+val_size), random_state=42)

# # # print(f'Training set size: {len(train_data)}')
# # # print(f'Validation set size: {len(val_data)}')
# # # print(f'Test set size: {len(test_data)}')

# # # # Define the features and target variable
# # # X_train = train_data[['Current (A)', 'Inside Temperature (°C)', 'Outside Temperature (°C)', 'Inside Humidity (%)']]
# # # y_train = train_data['Power (W)']

# # # X_val = val_data[['Current (A)', 'Inside Temperature (°C)', 'Outside Temperature (°C)', 'Inside Humidity (%)']]
# # # y_val = val_data['Power (W)']

# # # X_test = test_data[['Current (A)', 'Inside Temperature (°C)', 'Outside Temperature (°C)', 'Inside Humidity (%)']]
# # # y_test = test_data['Power (W)']

# # # # Create an XGBRegressor (tends to perform better than Random Forest for regression)
# # # regressor = XGBRegressor(n_estimators=200, max_depth=5, random_state=42)

# # # # Train the regressor
# # # regressor.fit(X_train, y_train)

# # # # Make predictions on the validation and test sets
# # # y_val_pred = regressor.predict(X_val)
# # # y_test_pred = regressor.predict(X_test)

# # # # Calculate Root Mean Squared Error
# # # val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
# # # test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
# # # print(f'Validation Root Mean Squared Error: {val_rmse}')
# # # print(f'Test Root Mean Squared Error: {test_rmse}')

# # # # Feature importance
# # # feature_importance = regressor.feature_importances_
# # # print('Feature Importance:', feature_importance)

# # # # Feature importance as a Pie Chart
# # # plt.figure(figsize=(8, 8))
# # # plt.pie(feature_importance, labels=X_train.columns, autopct='%1.1f%%', startangle=140)
# # # plt.title('Feature Importance')
# # # plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# # # plt.show()

# # # # Bar graph for dataset sizes
# # # sizes = [len(train_data), len(val_data), len(test_data)]
# # # labels = ['Training Set', 'Validation Set', 'Test Set']
# # # plt.bar(labels, sizes, color=['blue', 'orange', 'green'])
# # # plt.title('Dataset Sizes')
# # # plt.ylabel('Number of Samples')
# # # plt.show()

# # # # Bar graph for Root Mean Squared Error
# # # rmse_values = [val_rmse, test_rmse]
# # # labels = ['Validation RMSE', 'Test RMSE']
# # # plt.bar(labels, rmse_values, color=['purple', 'red'])
# # # plt.title('Root Mean Squared Error Comparison')
# # # plt.ylabel('Root Mean Squared Error')
# # # plt.ylim(0, max(rmse_values) * 1.2)  # Adjust ylim for better visualization
# # # plt.show()

# # # # Now let's find the ideal temperature and current to reduce power consumption
# # # temperature_range = np.arange(16, 31, 1)  # Adjust the temperature range as needed
# # # current_range = np.arange(5, 15, 1)  # Adjust the current range as needed
# # # power_consumption = []

# # # for temp in temperature_range:
# # #     for curr in current_range:
# # #         new_data = pd.DataFrame({
# # #             'Current (A)': [curr],
# # #             'Inside Temperature (°C)': [temp],
# # #             'Outside Temperature (°C)': [25],  # Assume a constant outside temperature
# # #             'Inside Humidity (%)': [50]  # Assume a constant inside humidity
# # #         })

# # #         predicted_power = regressor.predict(new_data)
# # #         power_consumption.append(predicted_power[0])

# # # # Find the optimal temperature and current to minimize power consumption
# # # optimal_temp, optimal_curr = np.unravel_index(np.argmin(power_consumption), (len(temperature_range), len(current_range)))
# # # optimal_temperature = temperature_range[optimal_temp]
# # # optimal_current = current_range[optimal_curr]
# # # min_power = min(power_consumption)

# # # print(f'Optimal Temperature: {optimal_temperature}°C')
# # # print(f'Optimal Current: {optimal_current} A')
# # # print(f'Minimum Power Consumption: {min_power} Watts')

# # # # Plot the power consumption vs. temperature and current
# # # fig, ax = plt.subplots(figsize=(10, 6))
# # # power_consumption = np.array(power_consumption).reshape(len(temperature_range), len(current_range))
# # # im = ax.imshow(power_consumption, cmap='hot', extent=[current_range.min(), current_range.max(), temperature_range.min(), temperature_range.max()], aspect='auto')
# # # plt.colorbar(im, ax=ax)
# # # plt.xlabel('Current (A)')
# # # plt.ylabel('Temperature (°C)')
# # # plt.title('Power Consumption vs. Temperature and Current')
# # # plt.show()

# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.metrics import mean_squared_error
# # from xgboost import XGBRegressor
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.preprocessing import MinMaxScaler
# # from sklearn import metrics

# # # Load the data from the CSV file
# # data = pd.read_csv('sensor_data.csv', encoding='latin1')

# # # Convert the 'Day' column to datetime
# # data['Day'] = pd.to_datetime(data['Day'])

# # # Extract hourly and daily data
# # hourly_data = data.copy()
# # daily_data = hourly_data.resample('D', on='Day').mean().reset_index()

# # # Normalize the data
# # scaler = MinMaxScaler()
# # hourly_data_normalized = scaler.fit_transform(hourly_data.drop(['Day'], axis=1))
# # daily_data_normalized = scaler.fit_transform(daily_data.drop(['Day'], axis=1))

# # # Plot all variables in one graph for hourly data (random)
# # plt.figure(figsize=(12, 8))
# # plt.plot(hourly_data.drop(['Day'], axis=1))
# # plt.legend(hourly_data.columns)
# # plt.title('Hourly Random Variables')
# # plt.xlabel('Time')
# # plt.ylabel('Value')
# # plt.xticks(rotation=45)
# # plt.show()

# # # Plot all variables in one graph for daily data (random)
# # plt.figure(figsize=(12, 8))
# # plt.plot(daily_data.drop(['Day'], axis=1))
# # plt.legend(daily_data.columns)
# # plt.title('Daily Random Variables')
# # plt.xlabel('Date')
# # plt.ylabel('Value')
# # plt.xticks(rotation=45)
# # plt.show()

# # # Plot all variables in one graph for hourly data (normalized)
# # plt.figure(figsize=(12, 8))
# # plt.plot(hourly_data_normalized)
# # plt.legend(hourly_data.columns)
# # plt.title('Hourly Normalized Variables')
# # plt.xlabel('Time')
# # plt.ylabel('Normalized Value')
# # plt.xticks(rotation=45)
# # plt.show()

# # # Plot all variables in one graph for daily data (normalized)
# # plt.figure(figsize=(12, 8))
# # plt.plot(daily_data_normalized)
# # plt.legend(daily_data.columns)
# # plt.title('Daily Normalized Variables')
# # plt.xlabel('Date')
# # plt.ylabel('Normalized Value')
# # plt.xticks(rotation=45)
# # plt.show()

# # # Plot temperature inside versus power using a hexbin plot
# # plt.figure(figsize=(10, 6))
# # plt.hexbin(hourly_data['Inside Temperature (°C)'], hourly_data['Power (W)'], gridsize=15, cmap='inferno')
# # plt.colorbar()
# # plt.title('Temperature Inside vs. Power')
# # plt.xlabel('Temperature Inside (°C)')
# # plt.ylabel('Power (W)')
# # plt.show()

# # # Split the data into training, validation, and testing sets
# # train_size = 0.7
# # val_size = 0.15
# # test_size = 0.15

# # train_data, other_data = train_test_split(hourly_data, train_size=train_size, random_state=42)
# # val_data, test_data = train_test_split(other_data, test_size=test_size/(test_size+val_size), random_state=42)

# # print(f'Training set size: {len(train_data)}')
# # print(f'Validation set size: {len(val_data)}')
# # print(f'Test set size: {len(test_data)}')

# # # Define the features and target variable
# # X_train = train_data[['Current (A)', 'Inside Temperature (°C)', 'Outside Temperature (°C)', 'Inside Humidity (%)']]
# # y_train = train_data['Power (W)']

# # X_val = val_data[['Current (A)', 'Inside Temperature (°C)', 'Outside Temperature (°C)', 'Inside Humidity (%)']]
# # y_val = val_data['Power (W)']

# # X_test = test_data[['Current (A)', 'Inside Temperature (°C)', 'Outside Temperature (°C)', 'Inside Humidity (%)']]
# # y_test = test_data['Power (W)']

# # # Create an XGBRegressor (tends to perform better than Random Forest for regression)
# # regressor = XGBRegressor(n_estimators=200, max_depth=5, random_state=42)

# # # Train the regressor
# # regressor.fit(X_train, y_train)

# # # Make predictions on the validation and test sets
# # y_val_pred = regressor.predict(X_val)
# # y_test_pred = regressor.predict(X_test)

# # # Calculate Root Mean Squared Error
# # val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
# # test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
# # print(f'Validation Root Mean Squared Error: {val_rmse}')
# # print(f'Test Root Mean Squared Error: {test_rmse}')

# # Feature importance
# feature_importance = regressor.feature_importances_
# print('Feature Importance:', feature_importance)


# # Feature importance as a Pie Chart
# plt.figure(figsize=(8, 8))
# plt.pie(feature_importance, labels=X_train.columns, autopct='%1.1f%%', startangle=140)
# plt.title('Feature Importance')
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.show()

# # # Bar graph for dataset sizes
# # sizes = [len(train_data), len(val_data), len(test_data)]
# # labels = ['Training Set', 'Validation Set', 'Test Set']
# # plt.bar(labels, sizes, color=['blue', 'orange', 'green'])
# # plt.title('Dataset Sizes')
# # plt.ylabel('Number of Samples')
# # plt.show()

# # # Bar graph for Root Mean Squared Error
# # rmse_values = [val_rmse, test_rmse]
# # labels = ['Validation RMSE', 'Test RMSE']
# # plt.bar(labels, rmse_values, color=['purple', 'red'])
# # plt.title('Root Mean Squared Error Comparison')
# # plt.ylabel('Root Mean Squared Error')
# # plt.ylim(0, max(rmse_values) * 1.2)  # Adjust ylim for better visualization
# # plt.show()

# # # Now let's find the ideal temperature and current to reduce power consumption
# # temperature_range = np.arange(16, 31, 1)  # Adjust the temperature range as needed
# # current_range = np.arange(5, 15, 1)  # Adjust the current range as needed
# # power_consumption = []

# # for temp in temperature_range:
# #     for curr in current_range:
# #         new_data = pd.DataFrame({
# #             'Current (A)': [curr],
# #             'Inside Temperature (°C)': [temp],
# #             'Outside Temperature (°C)': [25],  # Assume a constant outside temperature
# #             'Inside Humidity (%)': [50]  # Assume a constant inside humidity
# #         })

# #         predicted_power = regressor.predict(new_data)
# #         power_consumption.append(predicted_power[0])

# # # Find the optimal temperature and current to minimize power consumption
# # optimal_temp, optimal_curr = np.unravel_index(np.argmin(power_consumption), (len(temperature_range), len(current_range)))
# # optimal_temperature = temperature_range[optimal_temp]
# # optimal_current = current_range[optimal_curr]
# # min_power = min(power_consumption)

# # print(f'Optimal Temperature: {optimal_temperature}°C')
# # print(f'Optimal Current: {optimal_current} A')
# # print(f'Minimum Power Consumption: {min_power} Watts')

# # Plot the power consumption vs. temperature and current
# fig, ax = plt.subplots(figsize=(10, 6))
# power_consumption = np.array(power_consumption).reshape(len(temperature_range), len(current_range))
# im = ax.imshow(power_consumption, cmap='hot', extent=[current_range.min(), current_range.max(), temperature_range.min(), temperature_range.max()], aspect='auto')
# plt.colorbar(im, ax=ax)
# plt.xlabel('Current (A)')
# plt.ylabel('Temperature (°C)')
# plt.title('Power Consumption vs. Temperature and Current')
# plt.show()

# # # assuming y_test are your true values and predictions are the predicted values
# # accuracy = metrics.accuracy_score(y_test, predictions)
# # confusion_matrix = metrics.confusion_matrix(y_test, predictions)
# # roc_auc = metrics.roc_auc_score(y_test, predictions)
# # log_loss = metrics.log_loss(y_test, predictions)
# # mae = metrics.mean_absolute_error(y_test, predictions)
# # mse = metrics.mean_squared_error(y_test, predictions)
# # rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

# # print(f"Accuracy: {accuracy}")
# # print(f"Confusion Matrix: \n{confusion_matrix}")
# # print(f"ROC AUC: {roc_auc}")
# # print(f"Log Loss: {log_loss}")
# # print(f"Mean Absolute Error: {mae}")
# # print(f"Mean Squared Error: {mse}")
# # print(f"Root Mean Squared Error: {rmse}")

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D

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
hourly_data['Power Factor * Current'] = hourly_data['Power Factor'] * hourly_data['Current (A)']

# Plot all variables in one graph for hourly data (random)
plt.figure(figsize=(12, 8))
plt.plot(hourly_data.drop(['Day'], axis=1))
plt.legend(hourly_data.columns)
plt.title('Hourly Random Variables')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()

# Plot all variables in one graph for daily data (random)
plt.figure(figsize=(12, 8))
plt.plot(daily_data.drop(['Day'], axis=1))
plt.legend(daily_data.columns)
plt.title('Daily Random Variables')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()

# Plot all variables in one graph for hourly data (normalized)
plt.figure(figsize=(12, 8))
plt.plot(hourly_data_normalized)
plt.legend(hourly_data.columns)
plt.title('Hourly Normalized Variables')
plt.xlabel('Time')
plt.ylabel('Normalized Value')
plt.xticks(rotation=45)
plt.show()

# Plot all variables in one graph for daily data (normalized)
plt.figure(figsize=(12, 8))
plt.plot(daily_data_normalized)
plt.legend(daily_data.columns)
plt.title('Daily Normalized Variables')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.xticks(rotation=45)
plt.show()

# Split the data into training, validation, and testing sets
train_size = 0.7
val_size = 0.15
test_size = 0.15

train_data, other_data = train_test_split(hourly_data, train_size=train_size, random_state=42)
val_data, test_data = train_test_split(other_data, test_size=test_size/(test_size+val_size), random_state=42)

print(f'Training set size: {len(train_data)}')
print(f'Validation set size: {len(val_data)}')
print(f'Test set size: {len(test_data)}')

# Define the features and target variable
features = ['Current (A)', 'Inside Temperature (°C)', 'Outside Temperature (°C)', 'Inside Humidity (%)', 'Temperature Difference', 'Humidity Difference', 'Power Factor * Current']

X_train = train_data[features]
y_train = train_data['Power (W)']

X_val = val_data[features]
y_val = val_data['Power (W)']

X_test = test_data[features]
y_test = test_data['Power (W)']

# Create an XGBRegressor with increased depth and estimators
regressor = XGBRegressor(n_estimators=500, max_depth=10, random_state=42)

# Train the regressor
regressor.fit(X_train, y_train)

# Make predictions on the validation and test sets
y_val_pred = regressor.predict(X_val)
y_test_pred = regressor.predict(X_test)

# Calculate Root Mean Squared Error
val_rmse = np.sqrt(metrics.mean_squared_error(y_val, y_val_pred))
test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
print(f'Validation Root Mean Squared Error: {val_rmse}')
print(f'Test Root Mean Squared Error: {test_rmse}')

# Feature importance
feature_importance = regressor.feature_importances_
print('Feature Importance:', feature_importance)


# Bar graph for dataset sizes
sizes = [len(train_data), len(val_data), len(test_data)]
labels = ['Training Set', 'Validation Set', 'Test Set']
plt.bar(labels, sizes, color=['blue', 'orange', 'green'])
plt.title('Dataset Sizes')
plt.ylabel('Number of Samples')
plt.show()

# Bar graph for Root Mean Squared Error
rmse_values = [val_rmse, test_rmse]
labels = ['Validation RMSE', 'Test RMSE']
plt.bar(labels, rmse_values, color=['purple', 'red'])
plt.title('Root Mean Squared Error Comparison')
plt.ylabel('Root Mean Squared Error')
plt.ylim(0, max(rmse_values) * 1.2)  # Adjust ylim for better visualization
plt.show()

# Now let's find the ideal combination of features to reduce power consumption
def find_optimal_features(features, temperature_range, current_range):
    power_consumption = []
    optimal_features = []
    min_power = float('inf')

    for temp in temperature_range:
        for curr in current_range:
            new_data = pd.DataFrame({
                'Current (A)': [curr],
                'Inside Temperature (°C)': [temp],
                'Outside Temperature (°C)': [25],  # Assume a constant outside temperature
                'Inside Humidity (%)': [50],  # Assume a constant inside humidity
                'Temperature Difference': [25 - temp],
                'Humidity Difference': [50 - 50],
                'Power Factor * Current': [0.9 * curr]  # Assume a constant power factor of 0.9
            })

            predicted_power = regressor.predict(new_data[features])
            power_consumption.append(predicted_power[0])

            if predicted_power[0] < min_power:
                min_power = predicted_power[0]
                optimal_temp = temp
                optimal_curr = curr
                optimal_features = new_data[features].iloc[0].values

    return optimal_temp, optimal_curr, optimal_features, min_power

# Find the optimal combination of features to minimize power consumption
temperature_range = np.arange(16, 31, 1)  # Adjust the temperature range as needed
current_range = np.arange(5, 15, 1)  # Adjust the current range as needed

optimal_temp, optimal_curr, optimal_features, min_power = find_optimal_features(features, temperature_range, current_range)

print(f'Optimal Temperature: {optimal_temp}°C')
print(f'Optimal Current: {optimal_curr} A')
print(f'Optimal Features: {optimal_features}')
print(f'Minimum Power Consumption: {min_power} Watts')

# Plot the 2D graph of power consumption vs. temperature for the optimal current
plt.figure(figsize=(8, 6))
plt.plot(temperature_range, [regressor.predict(pd.DataFrame({
    'Current (A)': [optimal_curr],
    'Inside Temperature (°C)': [temp],
    'Outside Temperature (°C)': [25],
    'Inside Humidity (%)': [50],
    'Temperature Difference': [25 - temp],
    'Humidity Difference': [50 - 50],
    'Power Factor * Current': [0.9 * optimal_curr]
}, index=[0])[features])[0] for temp in temperature_range])
plt.xlabel('Temperature (°C)')
plt.ylabel('Power Consumption (W)')
plt.title('Power Consumption vs. Temperature')
plt.show()


# Plot the 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(current_range, temperature_range)
power_consumption = []

for temp in temperature_range:
    row = []
    for curr in current_range:
        new_data = pd.DataFrame({
            'Current (A)': [curr],
            'Inside Temperature (°C)': [temp],
            'Outside Temperature (°C)': [25],
            'Inside Humidity (%)': [50],
            'Temperature Difference': [25 - temp],
            'Humidity Difference': [50 - 50],
            'Power Factor * Current': [0.9 * curr]
        })
        predicted_power = regressor.predict(new_data[features])
        row.append(predicted_power[0])
    power_consumption.append(row)

power_consumption = np.array(power_consumption)

Z = power_consumption.reshape(len(temperature_range), len(current_range))

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('Current (A)')
ax.set_ylabel('Temperature (°C)')
ax.set_zlabel('Power Consumption (W)')
ax.set_title('Power Consumption vs. Temperature and Current')
plt.show()

# Plot the power consumption vs. temperature and current using imshow
fig, ax = plt.subplots(figsize=(10, 6))
power_consumption = np.array(power_consumption).reshape(len(temperature_range), len(current_range))
im = ax.imshow(power_consumption, cmap='hot', extent=[current_range.min(), current_range.max(), temperature_range.min(), temperature_range.max()], aspect='auto')
plt.colorbar(im, ax=ax)
plt.xlabel('Current (A)')
plt.ylabel('Temperature (°C)')
plt.title('Power Consumption vs. Temperature and Current')
plt.show()

# Plot the optimal feature selection
optimal_features_dict = dict(zip(features, optimal_features))
optimal_features_values = list(optimal_features_dict.values())
optimal_features_labels = list(optimal_features_dict.keys())

fig, ax = plt.subplots(figsize=(20, 15))  # Increase the size of the plot even more
wedges, texts, autotexts = ax.pie(optimal_features_values, autopct='%1.1f%%', startangle=90, labeldistance=1.5)  # Remove labels here
ax.axis('equal')  
ax.set_title('Optimal Feature Selection')

# Set individual labels
for i, (wedge, text, autotext) in enumerate(zip(wedges, texts, autotexts)):
    label = optimal_features_labels[i]
    text.set_text(label)  # Set the label
    text.set_fontsize(12)
    autotext.set_fontsize(12)

# Adjust the position of the legend outside the plot
plt.legend(wedges, optimal_features_labels, title="Features", loc="center left", bbox_to_anchor=(1.5, 0, 0.5, 1))

plt.show()

# Evaluate the model on the test set
y_test_pred = regressor.predict(X_test)

# Calculate evaluation metrics
mae = metrics.mean_absolute_error(y_test, y_test_pred)
mse = metrics.mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")