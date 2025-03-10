import os
import json  # Add missing json import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Add missing seaborn import
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
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
from tensorflow.keras.layers import Bidirectional, MultiHeadAttention, LayerNormalization, Dropout, Flatten
from scikeras.wrappers import KerasRegressor  # New import
from sklearn.base import BaseEstimator, RegressorMixin  # Add missing import


# Setup and Initialization
def setup_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_dir, 'application.log'), level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info("Script started.")
    return output_dir

# Data Preprocessing
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Try different encodings
    encodings = ['latin1', 'utf-8', 'cp1252']
    data = None
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    if data is None:
        raise ValueError("Could not read file with any of the attempted encodings")

    data['Day'] = pd.to_datetime(data['Day'])
    
    # Print column names to debug
    logging.info(f"Original columns: {data.columns.tolist()}")
    
    def normalize_column_name(col):
        # Remove special characters and normalize spaces
        col = col.strip()
        col = col.replace('Â', '')
        col = col.replace('°', '')
        col = col.replace('º', '')
        col = col.replace('ÂºC', 'C')
        col = col.replace('Â°C', 'C')
        col = col.replace('°C', 'C')
        col = col.replace('(C)', '(°C)')
        return col

    # Normalize all column names
    data.columns = [normalize_column_name(col) for col in data.columns]

    # Define expected column names
    numerical_features = [
        'Voltage (V)', 'Current (A)', 'Power Factor', 'Frequency (Hz)', 'Energy (kWh)',
        'Inside Temperature (°C)', 'Outside Temperature (°C)', 'Inside Humidity (%)', 'Outside Humidity (%)'
    ]
    
    # Create a comprehensive mapping for temperature columns
    temp_variations = [
        ('Inside Temperature(C)', 'Inside Temperature (°C)'),
        ('Outside Temperature(C)', 'Outside Temperature (°C)'),
        ('Inside Temperature (C)', 'Inside Temperature (°C)'),
        ('Outside Temperature (C)', 'Outside Temperature (°C)'),
        ('Inside Temperature', 'Inside Temperature (°C)'),
        ('Outside Temperature', 'Outside Temperature (°C)'),
        ('InsideTemp(C)', 'Inside Temperature (°C)'),
        ('OutsideTemp(C)', 'Outside Temperature (°C)'),
        ('Inside_Temperature', 'Inside Temperature (°C)'),
        ('Outside_Temperature', 'Outside Temperature (°C)')
    ]
    
    # Try to find and map temperature columns
    for old_name, new_name in temp_variations:
        if old_name in data.columns:
            data = data.rename(columns={old_name: new_name})
            logging.info(f"Renamed column '{old_name}' to '{new_name}'")
    
    # Function to find closest matching column
    def find_closest_match(target, columns):
        import difflib
        matches = difflib.get_close_matches(normalize_column_name(target), 
                                          [normalize_column_name(col) for col in columns], 
                                          n=1, cutoff=0.7)
        if matches:
            orig_cols = [col for col in columns if normalize_column_name(col) == matches[0]]
            return orig_cols[0] if orig_cols else None
        return None

    # Try to map required columns
    for required_col in numerical_features:
        if required_col not in data.columns:
            match = find_closest_match(required_col, data.columns)
            if match:
                data = data.rename(columns={match: required_col})
                logging.info(f"Mapped '{match}' to required column '{required_col}'")
            else:
                logging.warning(f"Could not find match for required column: {required_col}")

    # Verify columns after mapping
    missing_cols = [col for col in numerical_features if col not in data.columns]
    if missing_cols:
        logging.error(f"Missing columns after mapping: {missing_cols}")
        logging.error(f"Available columns: {data.columns.tolist()}")
        # Create missing columns with NaN values instead of raising error
        for col in missing_cols:
            data[col] = np.nan
            logging.warning(f"Created missing column '{col}' with NaN values")

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
    
    return data, feature_scaler, target_scaler

# Model Training and Evaluation
def train_lstm_model(X_train, y_train, features):
    # Build LSTM model
    def build_lstm_model():
        input_layer = Input(shape=(1, len(features)))
        lstm_1 = LSTM(64, return_sequences=True)(input_layer)
        lstm_2 = LSTM(32, activation='relu')(lstm_1)
        output_layer = Dense(1)(lstm_2)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        return model

    lstm_model = build_lstm_model()

    # Compile the model with the new optimizer
    optimizer = Adam(learning_rate=0.01)
    lstm_model.compile(optimizer=optimizer, loss='mse')

    # Define learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Use EarlyStopping to optimize execution time
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping, lr_scheduler], verbose=1)

    return lstm_model

def evaluate_model(model, X_test, y_test):
    # Generate predictions and evaluate
    predictions = model.predict(X_test)
    # Reshape predictions to match y_test dimensions
    predictions = predictions.reshape(-1)
    y_test = y_test.values if hasattr(y_test, 'values') else y_test
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    performance_metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
    
    return performance_metrics

# Anomaly Detection
def detect_anomalies(autoencoder, X, threshold):
    # Compute reconstruction error and detect anomalies
    X_pred = autoencoder.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=1)
    anomalies = mse > threshold
    return anomalies

# Visualization
def create_visualizations(data, output_dir):
    # Plot power consumption trends
    plt.figure()
    plt.plot(data['Day'], data['Power (W)'])
    plt.title('Power Consumption Over Time')
    plt.xlabel('Day')
    plt.ylabel('Power (W)')
    plt.savefig(os.path.join(output_dir, 'power_trends.png'))
    plt.close()

    # Plot anomalies
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

# Optimization
def predict_optimal_settings(lstm_model, autoencoder, temperature_range, current_range, threshold, feature_scaler, feature_columns, features):  # Add missing parameters
    logging.info("Optimal settings prediction started.")
    
    temp_grid, curr_grid = np.meshgrid(temperature_range, current_range)
    temp_values = temp_grid.flatten()
    curr_values = curr_grid.flatten()
    
    # Initialize new_data with the correct number of rows
    n_samples = len(temp_values)
    new_data = pd.DataFrame({
        'Voltage (V)': np.full(n_samples, 220.0),
        'Current (A)': curr_values,
        'Voltage (V)': np.full(n_samples, 220.0),
        'Current (A)': curr_values,
        'Power Factor': np.full(n_samples, 0.9),
        'Frequency (Hz)': np.full(n_samples, 50.0),
        'Energy (kWh)': np.zeros(n_samples),
        'Inside Temperature (°C)': temp_values,
        'Outside Temperature (°C)': np.full(n_samples, 25.0),
        'Inside Humidity (%)': np.full(n_samples, 50.0),
        'Outside Humidity (%)': np.full(n_samples, 50.0),
        'Temperature Difference': np.full(n_samples, 25.0) - temp_values,  # Calculate properly
        'Humidity Difference': np.zeros(n_samples),  # Will be calculated
        'Power Factor * Current': np.full(n_samples, 0.9) * curr_values,  # Calculate properly
        'Current Squared': curr_values ** 2  # Calculate properly
    })
    
    # Update calculated columns
    new_data['Temperature Difference'] = new_data['Outside Temperature (°C)'] - new_data['Inside Temperature (°C)']
    new_data['Humidity Difference'] = new_data['Outside Humidity (%)'] - new_data['Inside Humidity (%)']
    new_data['Power Factor * Current'] = new_data['Power Factor'] * new_data['Current (A)']
    new_data['Current Squared'] = new_data['Current (A)'] ** 2
    
    # Ensure column order matches feature_columns
    new_data = new_data[feature_columns]
    
    # Transform the data using the scaler
    try:
        new_data_normalized = feature_scaler.transform(new_data)
        new_data_normalized_df = pd.DataFrame(new_data_normalized, columns=feature_columns)
        
        # Prepare data for LSTM prediction
        lstm_input = new_data_normalized_df[features].values.reshape((len(new_data), 1, len(features)))
        
        # Rest of the function remains the same
        predicted_power_scaled = lstm_model.predict(lstm_input)
        predicted_power = target_scaler.inverse_transform(predicted_power_scaled).flatten()
        
        # Prepare data for autoencoder
        autoencoder_input = new_data_normalized_df[features].values
        reconstruction = autoencoder.predict(autoencoder_input)
        reconstruction_error = np.mean(np.power(autoencoder_input - reconstruction, 2), axis=1)
        
        # Create mask for valid conditions
        valid_mask = ((new_data['Inside Temperature (°C)'] >= 21) & 
                     (new_data['Inside Temperature (°C)'] <= 25) & 
                     (reconstruction_error < threshold) &
                     (new_data['Inside Humidity (%)'] >= 40) & 
                     (new_data['Inside Humidity (%)'] <= 60))
        
        # Find valid indices
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            # Find optimal settings among valid combinations
            valid_powers = predicted_power[valid_indices]
            min_power_idx = np.argmin(valid_powers)
            optimal_settings = new_data.iloc[valid_indices[min_power_idx]]
            min_power = predicted_power[valid_indices[min_power_idx]]
            optimal_temp = optimal_settings['Inside Temperature (°C)']
            logging.info(f"Optimal settings found at temperature {optimal_temp}°C with power consumption {min_power} W.")
        else:
            optimal_settings = None
            min_power = None
            optimal_temp = None
            logging.warning("No optimal settings found within the specified ranges.")
        
        logging.info("Optimal settings prediction completed.")
        
        return optimal_settings, min_power, optimal_temp
    
    except Exception as e:
        logging.error(f"Error during optimization: {str(e)}")
        return None, None, None

# Report Generation
def generate_report(metrics, optimal_settings, data, output_dir, cv_results=None, feature_importance=None):
    # Generate Markdown report summarizing outputs
    logging.info("Generating report.")
    try:
        with open(os.path.join(output_dir, 'report.md'), 'w') as report_file:
            report_file.write("# Analysis Report\n")
            report_file.write("## Model Performance\n")
            report_file.write(f"- Mean Absolute Error (MAE): {metrics['mae']:.4f}\n")
            report_file.write(f"- Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n")
            report_file.write(f"- R² Score: {metrics['r2']:.4f}\n")
            
            report_file.write("\n## Optimal Settings\n")
            if optimal_settings is not None:
                report_file.write(f"- **Optimal Temperature:** {optimal_settings['Inside Temperature (°C)']}°C\n")
                if 'min_power' in metrics:
                    report_file.write(f"- **Minimum Power Consumption:** {metrics['min_power']:.2f} W\n")
                report_file.write(f"- **Optimal Settings Details:**\n")
                report_file.write(f"```\n{optimal_settings}\n```\n")
            else:
                report_file.write("No optimal settings found within the specified ranges.\n")
            
            report_file.write("\n## Anomalies Detected\n")
            # Check for anomaly columns before accessing them
            autoencoder_anomalies = data.get('Anomaly_Autoencoder', pd.Series([0] * len(data)))
            isolation_forest_anomalies = data.get('Anomaly', pd.Series([0] * len(data)))
            
            report_file.write(f"- Total anomalies detected by Autoencoder: {autoencoder_anomalies.sum()}\n")
            report_file.write(f"- Total anomalies detected by Isolation Forest: {(isolation_forest_anomalies == -1).sum()}\n")
            
            if 'Cluster' in data.columns:
                report_file.write("\n## Cluster Analysis\n")
                report_file.write("- Cluster distribution:\n")
                cluster_counts = data['Cluster'].value_counts()
                for cluster_id, count in cluster_counts.items():
                    report_file.write(f"  - Cluster {cluster_id}: {count} instances\n")
            
            if cv_results:
                report_file.write("\n## Cross-Validation Results\n")
                for model_name, metrics in cv_results.items():
                    report_file.write(f"### {model_name}\n")
                    report_file.write(f"- MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}\n")
                    report_file.write(f"- RMSE: {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}\n")
                    report_file.write(f"- R²: {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}\n")
            
            if feature_importance is not None:
                report_file.write("\n## Feature Importance\n")
                report_file.write(feature_importance.to_string(index=False))
            
        logging.info("Report generated successfully.")
    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        raise

# Add these model architectures between the existing training functions:

def build_bi_lstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
        Bidirectional(LSTM(64, activation='relu')),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_attention_model(input_shape):
    input_layer = Input(shape=input_shape)
    attention_layer = MultiHeadAttention(num_heads=4, key_dim=8)(input_layer, input_layer)
    flatten_layer = Flatten()(attention_layer)
    dense_layer = Dense(64, activation='relu')(flatten_layer)
    output_layer = Dense(1)(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_combined_lstm_attention_model(input_shape):
    input_layer = Input(shape=input_shape)
    lstm_layer = LSTM(64, return_sequences=True)(input_layer)
    attention_layer = MultiHeadAttention(num_heads=4, key_dim=8)(lstm_layer, lstm_layer)
    attention_normalized = LayerNormalization()(attention_layer)
    dense_layer = Dense(64, activation='relu')(attention_normalized)
    dropout_layer = Dropout(0.2)(dense_layer)
    output_layer = Dense(1)(dropout_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(32, activation="relu")(input_layer)
    encoder = Dense(16, activation="relu")(encoder)
    encoder = Dropout(0.2)(encoder)
    decoder = Dense(32, activation="relu")(encoder)
    decoder = Dense(input_dim, activation="linear")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def perform_hyperparameter_tuning(X_train, y_train):
    def create_model(neurons=64, learning_rate=0.001):
        model = Sequential([
            LSTM(neurons, input_shape=(1, X_train.shape[2]), return_sequences=True),
            LSTM(neurons // 2, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model

    class CustomLSTMRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, neurons=64, learning_rate=0.001, epochs=50, batch_size=32, verbose=0):
            self.neurons = neurons
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.verbose = verbose
            self.model = None

        def fit(self, X, y):
            self.model = create_model(
                neurons=self.neurons,
                learning_rate=self.learning_rate
            )
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=self.verbose
            )
            return self

        def predict(self, X):
            if self.model is None:
                raise RuntimeError("Model has not been fitted yet.")
            return self.model.predict(X).reshape(-1)

        def score(self, X, y):
            predictions = self.predict(X)
            return -mean_squared_error(y, predictions)  # Negative MSE for scoring

    # Create model wrapper
    model = CustomLSTMRegressor()
    
    # Define parameter grid
    param_grid = {
        'neurons': [32, 64, 128],
        'learning_rate': [0.01, 0.001, 0.0001],
        'batch_size': [16, 32, 64]
    }
    
    # Create GridSearchCV with proper configuration
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=1
    )
    
    # Ensure input data is properly shaped
    X_train_reshaped = X_train if len(X_train.shape) == 3 else X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    
    # Perform grid search
    try:
        grid_result = grid.fit(X_train_reshaped, y_train)
        logging.info(f"Best parameters found: {grid_result.best_params_}")
        logging.info(f"Best score: {grid_result.best_score_}")
        return grid_result
    except Exception as e:
        logging.error(f"Hyperparameter tuning failed: {str(e)}")
        raise

def sanitize_filename(filename):
    """Remove or replace invalid characters in filenames"""
    # Map of invalid characters to their replacements
    invalid_chars = {
        '<': '_lt_',
        '>': '_gt_',
        ':': '_',
        '"': '_',
        '/': '_',
        '\\': '_',
        '|': '_',
        '?': '_',
        '*': '_x_',
        ' ': '_',
        '(': '_',
        ')': '_',
        '+': '_plus_',
        '°': 'deg',
        '%': 'pct'
    }
    
    # Replace each invalid character
    for char, replacement in invalid_chars.items():
        filename = filename.replace(char, replacement)
    
    # Remove any remaining invalid characters
    filename = ''.join(c for c in filename if c.isalnum() or c in '_-.')
    return filename

def perform_shap_analysis(model, X_train, X_test, features, output_dir):
    """Perform SHAP analysis with proper file naming."""
    logging.info("Starting SHAP analysis.")
    try:
        n_samples = 50
        X_sample = shap.sample(X_test, n_samples)
        X_train_summary = shap.kmeans(X_train, n_samples)
        
        def batch_predict(data):
            data_reshaped = data.reshape((data.shape[0], 1, data.shape[1]))
            return model.predict(data_reshaped).flatten()
        
        explainer = shap.KernelExplainer(batch_predict, X_train_summary)
        shap_values = explainer.shap_values(X_sample)
        
        # Generate SHAP plots with sanitized filenames
        for feature in features:
            feature_idx = features.index(feature)
            safe_feature_name = sanitize_filename(feature)
            
            try:
                shap.dependence_plot(
                    feature_idx, 
                    shap_values,
                    X_sample,
                    feature_names=features,
                    interaction_index=None,
                    show=False
                )
                plt.savefig(os.path.join(output_dir, f'shap_dependence_{safe_feature_name}.png'))
                plt.close()
            except Exception as e:
                logging.error(f"Error saving SHAP plot for feature {feature}: {str(e)}")
                plt.close()
        
        logging.info("SHAP analysis completed successfully.")
        return shap_values
    except Exception as e:
        logging.error(f"SHAP analysis failed: {str(e)}")
        raise

def detailed_shap_analysis(model, X_train, X_test, features, output_dir):
    """Detailed SHAP analysis with multiple visualizations"""
    try:
        # Initialize SHAP analysis
        n_samples = 50
        X_sample = shap.sample(X_test, n_samples)
        X_train_summary = shap.kmeans(X_train, n_samples)
        
        def batch_predict(data):
            data_reshaped = data.reshape((data.shape[0], 1, data.shape[1]))
            return model.predict(data_reshaped).flatten()
        
        # Create SHAP explainer and calculate values
        explainer = shap.KernelExplainer(batch_predict, X_train_summary)
        shap_values = explainer.shap_values(X_sample)
        
        # Generate basic SHAP plots with sanitized filenames
        for feature in features:
            feature_idx = features.index(feature)
            safe_feature_name = sanitize_filename(feature)
            
            try:
                shap.dependence_plot(
                    feature_idx, 
                    shap_values,
                    X_sample,
                    feature_names=features,
                    interaction_index=None,
                    show=False
                )
                plt.savefig(os.path.join(output_dir, f'shap_dependence_{safe_feature_name}.png'))
                plt.close()
            except Exception as e:
                logging.error(f"Error saving SHAP plot for feature {feature}: {str(e)}")
                plt.close()
        
        # Additional visualizations
        try:
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.savefig(os.path.join(output_dir, 'shap_summary_bar.png'))
            plt.close()
            
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.savefig(os.path.join(output_dir, 'shap_summary_dot.png'))
            plt.close()
        except Exception as e:
            logging.error(f"Error saving SHAP summary plots: {str(e)}")
            plt.close()
        
        # Feature importance ranking
        feature_importance = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame(list(zip(features, feature_importance)), 
                                   columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df, shap_values
        
    except Exception as e:
        logging.error(f"Detailed SHAP analysis failed: {str(e)}")
        raise

def build_advanced_lstm(features):  # Add features parameter
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(1, len(features))),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    return model

def build_complex_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    # Encoder
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = LayerNormalization()(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(32, activation='relu')(encoded)
    decoded = LayerNormalization()(decoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return autoencoder

def detailed_shap_analysis(model, X_train, X_test, features, output_dir):
    """Detailed SHAP analysis with multiple visualizations"""
    # Initialize SHAP analysis
    n_samples = 50
    X_sample = shap.sample(X_test, n_samples)
    X_train_summary = shap.kmeans(X_train, n_samples)
    
    def batch_predict(data):
        data_reshaped = data.reshape((data.shape[0], 1, data.shape[1]))
        return model.predict(data_reshaped).flatten()
    
    # Create SHAP explainer and calculate values
    explainer = shap.KernelExplainer(batch_predict, X_train_summary)
    shap_values = explainer.shap_values(X_sample)
    
    # Generate basic SHAP plots
    for feature in features:
        feature_idx = features.index(feature)
        shap.dependence_plot(
            feature_idx, 
            shap_values,
            X_sample,
            feature_names=features,
            interaction_index=None,
            show=False
        )
        plt.savefig(os.path.join(output_dir, f'shap_dependence_{sanitize_filename(feature)}.png'))
        plt.close()
    
    # Additional SHAP visualizations continue here...

    # Additional SHAP visualizations
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig(os.path.join(output_dir, 'shap_summary_bar.png'))
    plt.close()
    
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(os.path.join(output_dir, 'shap_summary_dot.png'))
    plt.close()
    
    # Feature importance ranking
    feature_importance = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame(list(zip(features, feature_importance)), 
                               columns=['feature', 'importance'])
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df, shap_values

def advanced_cross_validation(models, X, y, n_splits=5):
    """Time series aware cross-validation with proper data reshaping"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = {}
    
    for name, model in models.items():
        metrics = {'mae': [], 'rmse': [], 'r2': []}
        for train_idx, val_idx in tscv.split(X):
            try:
                # Get the split data
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                # Ensure proper reshaping based on model type
                if isinstance(model, Sequential) or isinstance(model, Model):
                    # For LSTM and other sequential models
                    if len(X_train_cv.shape) == 2:
                        X_train_cv = X_train_cv.reshape((X_train_cv.shape[0], 1, X_train_cv.shape[1]))
                        X_val_cv = X_val_cv.reshape((X_val_cv.shape[0], 1, X_val_cv.shape[1]))
                    
                    # Ensure y is properly shaped
                    y_train_cv = np.array(y_train_cv).reshape(-1, 1)
                    y_val_cv = np.array(y_val_cv).reshape(-1, 1)
                    
                    # Train the model
                    model.fit(
                        X_train_cv, 
                        y_train_cv,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_val_cv, y_val_cv),
                        verbose=0,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
                    )
                    
                    # Get predictions
                    predictions = model.predict(X_val_cv, verbose=0)
                    
                else:
                    # For non-sequential models
                    model.fit(X_train_cv, y_train_cv)
                    predictions = model.predict(X_val_cv)
                
                # Ensure predictions and y_val_cv are 1D for metric calculation
                predictions = predictions.ravel()
                y_val_cv = y_val_cv.ravel()
                
                # Calculate metrics
                metrics['mae'].append(mean_absolute_error(y_val_cv, predictions))
                metrics['rmse'].append(np.sqrt(mean_squared_error(y_val_cv, predictions)))
                metrics['r2'].append(r2_score(y_val_cv, predictions))
                
            except Exception as e:
                logging.error(f"Error in cross-validation for model {name}: {str(e)}")
                continue
        
        # Calculate mean and std of metrics if we have any successful folds
        if metrics['mae']:
            cv_results[name] = {
                'mae_mean': np.mean(metrics['mae']),
                'mae_std': np.std(metrics['mae']),
                'rmse_mean': np.mean(metrics['rmse']),
                'rmse_std': np.std(metrics['rmse']),
                'r2_mean': np.mean(metrics['r2']),
                'r2_std': np.std(metrics['r2'])
            }
        else:
            logging.warning(f"No successful cross-validation folds for model {name}")
            cv_results[name] = {
                'mae_mean': np.nan,
                'mae_std': np.nan,
                'rmse_mean': np.nan,
                'rmse_std': np.nan,
                'r2_mean': np.nan,
                'r2_std': np.nan
            }
    
    return cv_results

def build_advanced_models():
    """Build all advanced model architectures"""
    models = {
        'lstm': Sequential([
            LSTM(128, return_sequences=True, input_shape=(1, len(features))),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            LSTM(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1),
            Dense(16, activation='relu'),
            Dense(1)
        ]),
        'bi_lstm': Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(128, return_sequences=True)),
            Bidirectional(LSTM(64, activation='relu')),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ]),
        'attention': build_attention_model(input_shape),
        'combined': build_combined_lstm_attention_model(input_shape)
    }
    return models

def comprehensive_shap_analysis(model, X_train, X_test, features, output_dir):
    """Detailed SHAP analysis with multiple visualizations"""
    # ...existing SHAP code...
    
    # Additional SHAP visualizations and metrics
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig(os.path.join(output_dir, 'shap_summary_bar.png'))
    plt.close()
    
    # Feature importance matrix
    feature_importance = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importance,
        'abs_importance': np.abs(feature_importance)
    }).sort_values('abs_importance', ascending=False)
    
    return importance_df, shap_values

def enhanced_cross_validation(models, X, y, n_splits=5):
    """Time series aware cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = {}
    
    for name, model in models.items():
        metrics = {'mae': [], 'rmse': [], 'r2': []}
        for train_idx, val_idx in tscv.split(X):
            # ...existing cross-validation code...
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            X_train_cv = X_train_cv.reshape((X_train_cv.shape[0], 1, X_train_cv.shape[1]))
            X_val_cv = X_val_cv.reshape((X_val_cv.shape[0], 1, X_val_cv.shape[1]))
            
            model.fit(X_train_cv, y_train_cv, epochs=50, batch_size=32, 
                     validation_split=0.2, verbose=0,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
            
            predictions = model.predict(X_val_cv)
            metrics['mae'].append(mean_absolute_error(y_val_cv, predictions))
            metrics['rmse'].append(np.sqrt(mean_squared_error(y_val_cv, predictions)))
            metrics['r2'].append(r2_score(y_val_cv, predictions))
            
            cv_results[name] = {
                'mae_mean': np.mean(metrics['mae']),
                'mae_std': np.std(metrics['mae']),
                'rmse_mean': np.mean(metrics['rmse']),
                'rmse_std': np.std(metrics['rmse']),
                'r2_mean': np.mean(metrics['r2']),
                'r2_std': np.std(metrics['r2'])
            }
    
    return cv_results

def build_complex_autoencoder(input_dim):
    """Enhanced autoencoder with additional layers and normalization"""
    input_layer = Input(shape=(input_dim,))
    # Encoder
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = LayerNormalization()(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(32, activation='relu')(encoded)
    decoded = LayerNormalization()(decoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return autoencoder

def advanced_visualization(data, shap_values, importance_df, output_dir, features):
    """Create comprehensive visualizations with proper error handling"""
    try:
        # Power consumption patterns
        plt.figure(figsize=(12, 6))
        plt.plot(data['Day'], data['Power (W)'], label='Actual Power')
        plt.title('Power Consumption Pattern Analysis')
        plt.xlabel('Time')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'power_pattern.png'))
        plt.close()
        
        # Feature importance heatmap
        plt.figure(figsize=(10, 8))
        pivot_table = importance_df.pivot_table(values='importance', index='feature')
        sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Feature Importance Heatmap')
        plt.savefig(os.path.join(output_dir, 'feature_importance_heatmap.png'))
        plt.close()
        
        # SHAP interaction plots
        for feature in features:
            try:
                safe_feature_name = sanitize_filename(feature)
                plt.figure(figsize=(10, 6))
                feature_idx = features.index(feature)
                shap.dependence_plot(
                    feature_idx,
                    shap_values,
                    data[features],
                    show=False,
                    interaction_index=None
                )
                plt.title(f'SHAP Interaction Plot - {feature}')
                plt.savefig(os.path.join(output_dir, f'shap_interaction_{safe_feature_name}.png'))
                plt.close()
            except Exception as e:
                logging.error(f"Error saving SHAP interaction plot for {feature}: {str(e)}")
                plt.close()
        
        # Anomaly distribution
        if 'Reconstruction_error' in data.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(data['Reconstruction_error'], bins=50)
            plt.axvline(x=np.percentile(data['Reconstruction_error'], 95),
                       color='r', linestyle='--', label='Anomaly Threshold')
            plt.title('Reconstruction Error Distribution')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'anomaly_distribution.png'))
            plt.close()
            
    except Exception as e:
        logging.error(f"Error in advanced visualization: {str(e)}")
        plt.close()

# Main Execution Flow
def main():
    output_dir = setup_output_directory('New_output')
    data, feature_scaler, target_scaler = load_and_preprocess_data('sensor_data.csv')
    
    # Prepare features and target variable
    features = ['Current (A)', 'Inside Temperature (°C)', 'Temperature Difference',
                'Humidity Difference', 'Power Factor * Current', 'Current Squared']
    X = data[features]
    y = data['Power (W)']

    # Split data for supervised learning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape data for LSTM
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    lstm_model = train_lstm_model(X_train_lstm, y_train, features)
    performance_metrics = evaluate_model(lstm_model, X_test_lstm, y_test)

    # Train multiple model architectures
    bi_lstm_model = build_bi_lstm_model((1, len(features)))
    attention_model = build_attention_model((1, len(features)))
    combined_model = build_combined_lstm_attention_model((1, len(features)))
    autoencoder = build_autoencoder(len(features))

    # Train models
    models = {
        'bi_lstm': bi_lstm_model,
        'attention': attention_model,
        'combined': combined_model
    }
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    for name, model in models.items():
        model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, 
                    validation_split=0.2, callbacks=[early_stopping], verbose=1)
        metrics = evaluate_model(model, X_test_lstm, y_test)
        logging.info(f"{name} model metrics: {metrics}")
        
    # Perform hyperparameter tuning
    try:
        best_params = perform_hyperparameter_tuning(X_train_lstm, y_train)
        logging.info(f"Best hyperparameters: {best_params.best_params_}")
        
        # Create and train model with best parameters
        best_model = create_model(**best_params.best_params_)
        best_model.fit(
            X_train_lstm, 
            y_train,
            epochs=50,
            batch_size=best_params.best_params_['batch_size'],
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
    except Exception as e:
        logging.error(f"Error during hyperparameter tuning: {str(e)}")
        # Continue with default model if tuning fails
        best_model = lstm_model

    # Modified SHAP analysis section
    try:
        shap_values = perform_shap_analysis(lstm_model, X_train, X_test, features, output_dir)
        logging.info("SHAP analysis completed successfully")
    except Exception as e:
        logging.error(f"SHAP analysis failed: {str(e)}")
        shap_values = None
    
    # Initialize threshold with a default value
    threshold = 0.1  # Default threshold value
    
    # Modified anomaly detection section
    try:
        autoencoder = build_autoencoder(len(features))
        autoencoder.fit(X, X, epochs=50, batch_size=32, verbose=0)
        
        reconstruction_errors = autoencoder.predict(X)
        reconstruction_error = np.mean(np.power(X - reconstruction_errors, 2), axis=1)
        threshold = np.percentile(reconstruction_error, 95)  # Update threshold if successful
        anomalies = detect_anomalies(autoencoder, X, threshold)
        data['Anomaly'] = anomalies
        data['Reconstruction_error'] = reconstruction_error
        logging.info(f"Anomaly detection threshold set to: {threshold}")
        
    except Exception as e:
        logging.error(f"Anomaly detection failed: {str(e)}")
        logging.warning(f"Using default threshold value: {threshold}")
        anomalies = np.zeros(len(X))
        data['Anomaly'] = 0
        data['Reconstruction_error'] = np.zeros(len(X))

    # Also ensure feature_columns is defined before optimization
    feature_columns = features  # Use the same features list defined earlier

    # Now proceed with optimization
    temperature_range = np.arange(18, 30, 1)
    current_range = np.arange(5, 15, 0.5)
    optimal_settings, min_power, optimal_temp = predict_optimal_settings(
        lstm_model, autoencoder, temperature_range, current_range, threshold, 
        feature_scaler, feature_columns, features
    )

    # Update metrics dictionary with min_power
    performance_metrics['min_power'] = min_power

    # Add Anomaly_Autoencoder column before report generation
    data['Anomaly_Autoencoder'] = detect_anomalies(autoencoder, X, threshold)

    # Report generation
    generate_report(performance_metrics, optimal_settings, data, output_dir)
    logging.info("Script completed successfully.")

    # Build and train advanced models
    advanced_lstm = build_advanced_lstm(features)
    complex_autoencoder = build_complex_autoencoder(len(features))
    
    # Perform detailed SHAP analysis
    importance_df, shap_values = detailed_shap_analysis(lstm_model, X_train, X_test, features, output_dir)
    
    # Perform advanced cross-validation
    models = {
        'lstm': lstm_model,
        'advanced_lstm': advanced_lstm,
        'bi_lstm': bi_lstm_model,
        'attention': attention_model,
        'combined': combined_model
    }
    cv_results = advanced_cross_validation(models, X.values, y.values)
    
    # Update report generation to include new metrics
    generate_report(performance_metrics, optimal_settings, data, output_dir, 
                   cv_results=cv_results, feature_importance=importance_df)
    
    # Save models and results
    lstm_model.save(os.path.join(output_dir, 'lstm_model.keras'))
    advanced_lstm.save(os.path.join(output_dir, 'advanced_lstm_model.keras'))
    complex_autoencoder.save(os.path.join(output_dir, 'complex_autoencoder_model.keras'))
    
    # Save feature importance results
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Save cross-validation results
    with open(os.path.join(output_dir, 'cross_validation_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=4)
    
    # Create final visualizations
    advanced_visualization(data, shap_values, importance_df, output_dir, features)
    
    # Save final processed data
    data.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
    
    # Final logging
    logging.info("All analyses completed successfully")
    logging.info(f"Results saved to {output_dir}")
    
    return {

        'models': models,
        'performance_metrics': performance_metrics,
        'cv_results': cv_results,
        'feature_importance': importance_df,
        'optimal_settings': optimal_settings,
        'shap_values': shap_values
    }


if __name__ == "__main__":
    main()

