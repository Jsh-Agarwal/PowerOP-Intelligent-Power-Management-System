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
import logging
from multiprocessing import Pool
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau


# Create the output directory if it doesn't exist
output_dir = 'New_output'
os.makedirs(output_dir, exist_ok=True)

# Setup logging
logging.basicConfig(filename=os.path.join(output_dir, 'application.log'), level=logging.INFO, format='%(asctime)s %(message)s')
logging.info("Script started.")

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
    
    # Evaluate loaded model performance
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    lstm_predictions = lstm_model.predict(X_test_lstm)
    mae = mean_absolute_error(y_test, lstm_predictions)
    mse = mean_squared_error(y_test, lstm_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, lstm_predictions)
    
    # Save evaluation metrics
    with open(os.path.join(output_dir, 'model_performance.txt'), 'w') as f:
        f.write(f"LSTM Model Performance (Loaded Model):\n")
        f.write(f"Mean Absolute Error: {mae}\n")
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Root Mean Squared Error: {rmse}\n")
        f.write(f"R^2 Score: {r2}\n")
else:
    # Reshape data for LSTM model
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Build and train LSTM model
    lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[1])),
    LSTM(32, activation='relu'),
    Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mse')

    
    # Compile the model with the new optimizer
    optimizer = Adam(learning_rate=0.01)
    lstm_model.compile(optimizer=optimizer, loss='mse')

    # Define learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Use EarlyStopping to optimize execution time
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping, lr_scheduler], verbose=1)

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
    encoder = Dense(32, activation="relu")(input_layer)
    encoder = Dense(16, activation="relu")(encoder)
    decoder = Dense(32, activation="relu")(encoder)
    decoder = Dense(input_dim, activation="linear")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    try:
        # Add EarlyStopping to autoencoder training
        early_stopping_autoencoder = EarlyStopping(monitor='val_loss', patience=3)
        history = autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.2,
                                  callbacks=[early_stopping_autoencoder], verbose=0)
        logging.info("Autoencoder training completed.")
    except Exception as e:
        logging.error(f"Autoencoder training failed: {e}")
        raise

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

# Define lstm_predict function
def lstm_predict(data):
    data_reshaped = data.reshape((data.shape[0], 1, data.shape[1]))
    predictions = lstm_model.predict(data_reshaped)
    return predictions.flatten()

# SHAP Analysis
logging.info("Starting SHAP analysis.")
try:
    # Take a smaller sample for computational efficiency
    n_samples = 50
    X_sample = shap.sample(X_test, n_samples)
    X_train_summary = shap.kmeans(X_train, n_samples)
    
    # Reshape data for LSTM input
    def batch_predict(data):
        data_reshaped = data.reshape((data.shape[0], 1, data.shape[1]))
        return lstm_model.predict(data_reshaped).flatten()
    
    # Create explainer with the batch prediction function
    explainer = shap.KernelExplainer(batch_predict, X_train_summary)
    shap_values = explainer.shap_values(X_sample)
    
    # Ensure proper shape of SHAP values
    if isinstance(shap_values, list):
        shap_values_array = shap_values[0]
    else:
        shap_values_array = shap_values
        
    # Ensure both arrays have matching dimensions
    if shap_values_array.ndim == 1:
        shap_values_array = shap_values_array.reshape(X_sample.shape[0], -1)
    
    # Verify shapes match before plotting
    assert shap_values_array.shape[0] == X_sample.shape[0], "Shape mismatch between SHAP values and features"
    
    def sanitize_filename(name):
        # Replace invalid characters with underscores
        invalid_chars = '<>:"/\\|?* '
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name
    
    # Generate SHAP plots
    for feature in features:
        feature_idx = features.index(feature)
        shap.dependence_plot(
            feature_idx, 
            shap_values_array,
            X_sample,
            feature_names=features,
            interaction_index=None,
            show=False
        )
        safe_filename = sanitize_filename(f'shap_dependence_{feature}')
        plt.savefig(os.path.join(output_dir, f'{safe_filename}.png'))
        plt.close()
    
    logging.info("SHAP analysis completed successfully.")
except Exception as e:
    logging.error(f"SHAP analysis failed: {e}")
    raise

# SHAP Dependence Plot for 'Current (A)'
shap.dependence_plot('Current (A)', shap_values_array, X_sample, feature_names=features, interaction_index=None, show=False)
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
logging.info("Creating visualizations.")
try:
    plt.figure(figsize=(10, 6))
    plt.plot(data['Day'], data['Power (W)'], label='Power Consumption')
    anomalies = data[data['Anomaly'] == -1]
    plt.scatter(anomalies['Day'], anomalies['Power (W)'], color='red', label='Anomalies')
    plt.xlabel('Day')
    plt.ylabel('Power (W)')
    plt.title('Power Consumption Over Time with Anomalies')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'power_trends_with_anomalies.png'))
    plt.close()
    
    # Cluster Visualization
    plt.figure()
    plt.scatter(data['Current (A)'], data['Power (W)'], c=data['Cluster'], cmap='viridis')
    plt.xlabel('Current (A)')
    plt.ylabel('Power (W)')
    plt.title('Clusters in Power Usage')
    plt.savefig(os.path.join(output_dir, 'clusters.png'))
    plt.close()
    logging.info("Visualizations completed.")
except Exception as e:
    logging.error(f"Visualization failed: {e}")
    raise

# Save the LSTM model
lstm_model.save(os.path.join(output_dir, 'lstm_model.keras'))

# Save the autoencoder model
autoencoder.save(os.path.join(output_dir, 'autoencoder_model.keras'))

# Save processed data
data.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)

# Function to predict optimal AC settings
def predict_optimal_settings(lstm_model, autoencoder, temperature_range, current_range, threshold):
    logging.info("Optimal settings prediction started.")
    
    # Batch generate all combinations
    temp_grid, curr_grid = np.meshgrid(temperature_range, current_range)
    temp_values = temp_grid.flatten()
    curr_values = curr_grid.flatten()
    
    # Create DataFrame for all combinations
    new_data = pd.DataFrame({
        'Voltage (V)': 220.0,
        'Current (A)': curr_values,
        'Power Factor': 0.9,
        'Frequency (Hz)': 50.0,
        'Inside Temperature (°C)': temp_values,
        'Outside Temperature (°C)': 25.0,
        'Inside Humidity (%)': 50.0,
        'Outside Humidity (%)': 50.0,
        'Energy (kWh)': 0.0,
        'Power (W)': 0.0
    })
    
    # Compute engineered features
    new_data['Temperature Difference'] = new_data['Outside Temperature (°C)'] - new_data['Inside Temperature (°C)']
    new_data['Humidity Difference'] = new_data['Outside Humidity (%)'] - new_data['Inside Humidity (%)']
    new_data['Power Factor * Current'] = new_data['Power Factor'] * new_data['Current (A)']
    new_data['Current Squared'] = new_data['Current (A)'] ** 2
    
    # Normalize features
    new_data_normalized = feature_scaler.transform(new_data[feature_columns])
    new_data_normalized_df = pd.DataFrame(new_data_normalized, columns=feature_columns)
    
    # Prepare data for LSTM prediction
    lstm_input = new_data_normalized_df[features].values.reshape((len(new_data), 1, len(features)))
    
    # Batch predict power consumption
    predicted_power_scaled = lstm_model.predict(lstm_input)
    predicted_power = target_scaler.inverse_transform(predicted_power_scaled).flatten()
    
    # Prepare data for autoencoder
    autoencoder_input = new_data_normalized_df[features].values
    
    # Batch prediction for autoencoder
    reconstruction = autoencoder.predict(autoencoder_input)
    reconstruction_error = np.mean(np.power(autoencoder_input - reconstruction, 2), axis=1)
    
    # Filter results based on threshold and find optimal settings
    valid_indices = reconstruction_error < threshold
    valid_powers = predicted_power[valid_indices]
    
    if len(valid_powers) > 0:
        min_power_idx = np.argmin(valid_powers)
        optimal_settings = new_data.iloc[valid_indices].iloc[min_power_idx]
        min_power = valid_powers[min_power_idx]
        optimal_temp = optimal_settings['Inside Temperature (°C)']
        logging.info(f"Optimal settings found at temperature {optimal_temp}°C with power consumption {min_power} W.")
    else:
        optimal_settings = None
        min_power = None
        optimal_temp = None
        logging.warning("No optimal settings found within the specified ranges.")
    
    logging.info("Optimal settings prediction completed.")
    
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

# Generate Markdown report
logging.info("Generating report.")
try:
    with open(os.path.join(output_dir, 'report.md'), 'w') as report_file:
        report_file.write("# Analysis Report\n")
        report_file.write("## Model Performance\n")
        report_file.write(f"- Mean Absolute Error (MAE): {mae:.4f}\n")
        report_file.write(f"- Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        report_file.write(f"- R² Score: {r2:.4f}\n")
        
        report_file.write("\n## Optimal Settings\n")
        if optimal_settings is not None:
            report_file.write(f"- **Optimal Temperature:** {optimal_temp}°C\n")
            report_file.write(f"- **Minimum Power Consumption:** {min_power:.2f} W\n")
            report_file.write(f"- **Optimal Settings Details:**\n")
            report_file.write(f"```\n{optimal_settings}\n```\n")
        else:
            report_file.write("No optimal settings found within the specified ranges.\n")
        
        report_file.write("\n## Anomalies Detected\n")
        report_file.write(f"- Total anomalies detected by Autoencoder: {data['Anomaly_Autoencoder'].sum()}\n")
        report_file.write(f"- Total anomalies detected by Isolation Forest: {(data['Anomaly'] == -1).sum()}\n")
        
        report_file.write("\n## Cluster Analysis\n")
        report_file.write("- Cluster distribution:\n")
        cluster_counts = data['Cluster'].value_counts()
        for cluster_id, count in cluster_counts.items():
            report_file.write(f"  - Cluster {cluster_id}: {count} instances\n")
    logging.info("Report generated.")
except Exception as e:
    logging.error(f"Report generation failed: {e}")
    raise

logging.info("Script completed successfully.")
