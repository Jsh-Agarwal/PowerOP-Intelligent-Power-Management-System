import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import shap

# Create the output directory if it doesn't exist
output_dir = 'New_output'
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess data
data = pd.read_csv('sensor_data.csv', encoding='latin1')
data['Day'] = pd.to_datetime(data['Day'])

# Handle missing values
numerical_features = ['Voltage (V)', 'Current (A)', 'Power Factor',
                      'Frequency (Hz)', 'Energy (kWh)', 'Inside Temperature (°C)',
                      'Outside Temperature (°C)', 'Inside Humidity (%)', 'Outside Humidity (%)']

# Use a separate scaler for the target variable
target_scaler = StandardScaler()
data['Power (W)'] = target_scaler.fit_transform(data[['Power (W)']])

# Impute missing values
imputer = KNNImputer(n_neighbors=5)
data[numerical_features] = imputer.fit_transform(data[numerical_features])

# Feature Engineering
data['Temperature Difference'] = data['Outside Temperature (°C)'] - data['Inside Temperature (°C)']
data['Humidity Difference'] = data['Outside Humidity (%)'] - data['Inside Humidity (%)']
data['Power Factor * Current'] = data['Power Factor'] * data['Current (A)']
data['Current Squared'] = data['Current (A)'] ** 2

# Normalize input features
feature_scaler = StandardScaler()
feature_columns = numerical_features + ['Temperature Difference', 'Humidity Difference', 'Power Factor * Current', 'Current Squared']
data[feature_columns] = feature_scaler.fit_transform(data[feature_columns])

# Prepare features and target variable
features = ['Current (A)', 'Inside Temperature (°C)', 'Temperature Difference',
            'Humidity Difference', 'Power Factor * Current', 'Current Squared']
X = data[features]
y = data['Power (W)']

# Split data for supervised learning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if models exist
lstm_model_path = os.path.join(output_dir, 'lstm_model.keras')
autoencoder_model_path = os.path.join(output_dir, 'autoencoder_model.keras')

if os.path.exists(lstm_model_path) and os.path.exists(autoencoder_model_path):
    # Load existing models
    lstm_model = load_model(lstm_model_path)
    autoencoder = load_model(autoencoder_model_path)
else:
    # Reshape data for LSTM model
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Build and train LSTM model
    lstm_model = Sequential([
        LSTM(50, input_shape=(1, X_train.shape[1]), activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')

    # Use EarlyStopping to optimize execution time
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    # Evaluate the model
    lstm_predictions = lstm_model.predict(X_test_lstm)
    mae = mean_absolute_error(y_test, lstm_predictions)
    mse = mean_squared_error(y_test, lstm_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, lstm_predictions)

    # Save evaluation metrics
    with open(os.path.join(output_dir, 'model_performance.txt'), 'w') as f:
        f.write(f"LSTM Model Performance:\n")
        f.write(f"Mean Absolute Error: {mae}\n")
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Root Mean Squared Error: {rmse}\n")
        f.write(f"R^2 Score: {r2}\n")

    # Train Autoencoder for anomaly detection
    input_dim = X.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(16, activation="relu")(input_layer)
    encoder = Dense(8, activation="relu")(encoder)
    decoder = Dense(8, activation="relu")(encoder)
    decoder = Dense(input_dim, activation="linear")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    history = autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Save the models
    lstm_model.save(lstm_model_path)
    autoencoder.save(autoencoder_model_path)

# Compute reconstruction error
X_pred = autoencoder.predict(X)
mse = np.mean(np.power(X - X_pred, 2), axis=1)
data['Reconstruction_error'] = mse

# Set threshold for anomalies (e.g., 95th percentile)
threshold = np.percentile(mse, 95)
data['Anomaly_Autoencoder'] = mse > threshold

# Unsupervised Learning - Anomaly detection with Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
data['Anomaly'] = iso_forest.fit_predict(X)

# Clustering with KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Save anomaly detection results
data[['Day', 'Reconstruction_error', 'Anomaly_Autoencoder']].to_csv(os.path.join(output_dir, 'anomaly_detection_results.csv'), index=False)

# Plot reconstruction error with anomalies
plt.figure(figsize=(10, 6))
plt.plot(data['Day'], data['Reconstruction_error'], label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel('Day')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error over Time')
plt.legend()
plt.savefig(os.path.join(output_dir, 'reconstruction_error_plot.png'))
plt.close()

# SHAP Analysis
# Reduce dataset size for SHAP to optimize computation time
X_sample = shap.sample(X_test, 50)
X_train_summary = shap.sample(X_train, 100)

# Create a function to wrap model prediction for SHAP
def lstm_predict(data):
    data_reshaped = data.reshape((data.shape[0], 1, data.shape[1]))
    return lstm_model.predict(data_reshaped)

explainer = shap.KernelExplainer(lstm_predict, X_train_summary)
shap_values = explainer.shap_values(X_sample, nsamples=100)

# Ensure the shapes match
shap_values = np.array(shap_values).reshape(len(X_sample), -1)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_sample, feature_names=features, show=False)
plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'))
plt.close()

# SHAP Dependence Plot for 'Current (A)'
shap.dependence_plot('Current (A)', shap_values, X_sample, feature_names=features, show=False)
plt.savefig(os.path.join(output_dir, 'shap_dependence_plot_current.png'))
plt.close()

# Visualizations
# Energy Trends
plt.figure()
plt.plot(data['Day'], data['Power (W)'])
plt.title('Power Consumption Over Time')
plt.xlabel('Day')
plt.ylabel('Power (W)')
plt.savefig(os.path.join(output_dir, 'power_trends.png'))
plt.close()

# Anomalies
anomalies = data[data['Anomaly'] == -1]
plt.figure()
plt.plot(data['Day'], data['Power (W)'], label='Normal')
plt.scatter(anomalies['Day'], anomalies['Power (W)'], color='red', label='Anomaly')
plt.legend()
plt.title('Anomalies in Power Consumption')
plt.xlabel('Day')
plt.ylabel('Power (W)')
plt.savefig(os.path.join(output_dir, 'anomalies.png'))
plt.close()

# Clustering Visualization
plt.figure()
plt.scatter(data['Current (A)'], data['Power (W)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Current (A)')
plt.ylabel('Power (W)')
plt.title('Clusters in Power Usage')
plt.savefig(os.path.join(output_dir, 'clusters.png'))
plt.close()

# Save the LSTM model
lstm_model.save(os.path.join(output_dir, 'lstm_model.keras'))

# Save the autoencoder model
autoencoder.save(os.path.join(output_dir, 'autoencoder_model.keras'))

# Save processed data
data.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)

# Function to predict optimal AC settings
def predict_optimal_settings(lstm_model, autoencoder, temperature_range, current_range, threshold):
    min_power = float('inf')
    optimal_settings = None
    optimal_temp = None

    # Combine numerical and engineered features
    all_features = numerical_features + ['Temperature Difference', 'Humidity Difference', 'Power Factor * Current', 'Current Squared']

    lstm_features = ['Current (A)', 'Inside Temperature (°C)', 'Temperature Difference',
                     'Humidity Difference', 'Power Factor * Current', 'Current Squared']

    for temp in temperature_range:
        for curr in current_range:
            # Create a DataFrame with all features used during scaling
            new_data = pd.DataFrame(columns=all_features)

            # Initialize with zeros and set correct data types
            for feature in all_features:
                new_data.loc[0, feature] = 0.0

            # Set known values
            new_data.loc[0, 'Voltage (V)'] = 220.0
            new_data.loc[0, 'Current (A)'] = curr
            new_data.loc[0, 'Power Factor'] = 0.9
            new_data.loc[0, 'Frequency (Hz)'] = 50.0
            new_data.loc[0, 'Inside Temperature (°C)'] = temp
            new_data.loc[0, 'Outside Temperature (°C)'] = 25.0
            new_data.loc[0, 'Inside Humidity (%)'] = 50.0
            new_data.loc[0, 'Outside Humidity (%)'] = 50.0
            new_data.loc[0, 'Energy (kWh)'] = 0.0
            new_data.loc[0, 'Power (W)'] = 0.0

            # Compute engineered features
            new_data.loc[0, 'Temperature Difference'] = new_data.loc[0, 'Outside Temperature (°C)'] - new_data.loc[0, 'Inside Temperature (°C)']
            new_data.loc[0, 'Humidity Difference'] = new_data.loc[0, 'Outside Humidity (%)'] - new_data.loc[0, 'Inside Humidity (%)']
            new_data.loc[0, 'Power Factor * Current'] = new_data.loc[0, 'Power Factor'] * new_data.loc[0, 'Current (A)']
            new_data.loc[0, 'Current Squared'] = new_data.loc[0, 'Current (A)'] ** 2

            # Reorder columns to match the scaler's expectation
            new_data = new_data[all_features]

            # Normalize the new data
            new_data_normalized = feature_scaler.transform(new_data)

            # Convert normalized data back to DataFrame
            new_data_normalized_df = pd.DataFrame(new_data_normalized, columns=all_features)

            # Prepare data for LSTM model
            lstm_input = new_data_normalized_df[lstm_features].values.reshape((1, 1, len(lstm_features)))

            # Predict power consumption
            predicted_power_scaled = lstm_model.predict(lstm_input)[0][0]
            predicted_power = target_scaler.inverse_transform([[predicted_power_scaled]])[0][0]

            # Prepare data for autoencoder
            autoencoder_input = new_data_normalized_df[lstm_features].values

            # Check for anomalies using autoencoder
            reconstruction_error = np.mean(np.power(autoencoder_input - autoencoder.predict(autoencoder_input), 2))

            if reconstruction_error < threshold and predicted_power < min_power:
                min_power = predicted_power
                optimal_settings = new_data.iloc[0]
                optimal_temp = temp

    return optimal_settings, min_power, optimal_temp

# Define the range of temperatures and currents to search for optimal settings
temperature_range = np.arange(18, 30, 1)  # Example range from 18°C to 30°C
current_range = np.arange(5, 15, 0.5)  # Example range from 5A to 15A

# Predict optimal settings
optimal_settings, min_power, optimal_temp = predict_optimal_settings(lstm_model, autoencoder, temperature_range, current_range, threshold)

# Save the optimal settings to a file
with open(os.path.join(output_dir, 'optimal_settings.txt'), 'w') as f:
    f.write(f"Optimal Settings for Minimum Power Consumption:\n")
    f.write(f"{optimal_settings}\n")
    f.write(f"Optimal Room Temperature: {optimal_temp}°C\n")
    f.write(f"Minimum Power Consumption: {min_power} W\n")