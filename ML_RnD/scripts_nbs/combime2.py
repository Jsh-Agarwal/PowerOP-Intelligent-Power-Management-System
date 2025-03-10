import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List, Union

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Flatten, Bidirectional, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import shap
import holidays
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Define all classes and functions from the individual files

class ExplainableAI:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.explainers = {}
        self.shap_values = {}

    def _reshape_input_data(self, data: np.ndarray, model_type: str) -> np.ndarray:
        if model_type == 'lstm':
            return data.reshape((data.shape[0], 1, -1))
        return data

    def _create_model_wrapper(self, model, model_type: str):
        def predict_wrapper(data):
            reshaped_data = self._reshape_input_data(data, model_type)
            return model.predict(reshaped_data)
        return predict_wrapper

    def create_explainer(self, model, X_train: np.ndarray, model_type: str, n_samples: int = 50) -> None:
        X_background = shap.sample(X_train, n_samples)
        predict_fn = self._create_model_wrapper(model, model_type)
        
        if model_type in ['lstm', 'autoencoder']:
            self.explainers[model_type] = shap.DeepExplainer(predict_fn, X_background)
        else:
            self.explainers[model_type] = shap.KernelExplainer(predict_fn, X_background)

    def compute_shap_values(self, X_test: np.ndarray, model_type: str, n_samples: int = 50) -> np.ndarray:
        if model_type not in self.explainers:
            raise ValueError(f"No explainer found for {model_type}")
        X_sample = shap.sample(X_test, n_samples)
        self.shap_values[model_type] = self.explainers[model_type].shap_values(X_sample)
        return self.shap_values[model_type]

    def generate_feature_importance(self, model_type: str, features: list) -> pd.DataFrame:
        if model_type not in self.shap_values:
            raise ValueError(f"No shap values computed for {model_type}")
        shap_vals = self.shap_values[model_type]
        importance = np.abs(shap_vals).mean(0)
        return pd.DataFrame({
            'feature': features,
            'importance': importance,
            'abs_importance': np.abs(importance)
        }).sort_values('abs_importance', ascending=False)

    def plot_shap_summary(self, model_type: str, features: list, X_test: np.ndarray) -> None:
        try:
            shap.summary_plot(self.shap_values.get(model_type), X_test, feature_names=features, show=False)
            plt.savefig(f"{self.output_dir}/shap_summary_{model_type}.png")
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting shap summary: {e}")

    def plot_feature_dependence(self, model_type: str, features: list, X_test: np.ndarray) -> None:
        try:
            # Plot dependence for first feature as an example.
            shap.dependence_plot(features[0], self.shap_values.get(model_type), X_test, show=False)
            plt.savefig(f"{self.output_dir}/shap_dependence_{features[0]}.png")
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting feature dependence: {e}")

class ModelBuilder:
    @staticmethod
    def build_basic_lstm(input_shape, output_dim=12):
        # Minimal implementation
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape))
        model.add(Dense(output_dim))
        model.compile(optimizer=Adam(), loss='mse')
        return model

    @staticmethod
    def build_advanced_lstm(input_shape, output_dim=12):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_dim))
        model.compile(optimizer=Adam(), loss='mse')
        return model

    @staticmethod
    def build_bi_lstm(input_shape, output_dim=12):
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dense(output_dim))
        model.compile(optimizer=Adam(), loss='mse')
        return model

    @staticmethod
    def build_attention(input_shape, output_dim=12):
        inputs = Input(shape=input_shape)
        lstm_out = LSTM(128, return_sequences=True)(inputs)
        attention_out = MultiHeadAttention(num_heads=2, key_dim=2)(lstm_out, lstm_out)
        lstm_out = LSTM(64)(attention_out)
        output = Dense(output_dim)(lstm_out)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(), loss='mse')
        return model

    @staticmethod
    def build_combined_lstm_attention(input_shape, output_dim=12):
        return ModelBuilder.build_attention(input_shape, output_dim)

    @staticmethod
    def build_model(input_shape):
        return ModelBuilder.build_basic_lstm(input_shape)

    @staticmethod
    def build_autoencoder(input_dim, encoding_dim=16):
        inputs = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(inputs)
        decoded = Dense(input_dim, activation='linear')(encoded)
        autoencoder = Model(inputs=inputs, outputs=decoded)
        autoencoder.compile(optimizer=Adam(), loss='mse')
        return autoencoder

class ModelTrainer:
    def __init__(self, input_shape, output_dim=12, sequence_length=24, model_dir=None):
        if len(input_shape) == 2:
            self.input_shape = (input_shape[0], input_shape[1])
        elif len(input_shape) == 3:
            self.input_shape = (input_shape[1], input_shape[2])
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")
        
        self.output_dim = output_dim
        self.models = {
            'bi_lstm': ModelBuilder.build_bi_lstm(self.input_shape, self.output_dim),
            'attention': ModelBuilder.build_attention(self.input_shape, self.output_dim),
            'combined': ModelBuilder.build_combined_lstm_attention(self.input_shape, self.output_dim)
        }
        self.sequence_length = sequence_length
        self.reshape_required = True
        self.model_dir = model_dir or Path("D:/PowerAmp/models")  # Common directory for models
        self.pretrained_model_dir = Path("D:/PowerAmp/outputs/analysis_20250215_175946/models")  # Pretrained models directory

    def compile_models(self, learning_rate=0.0005):  # Lower learning rate
        for name, model in self.models.items():
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']  # Changed metric to MAE
            )

    def load_existing_models(self):
        if self.pretrained_model_dir:
            for name in self.models.keys():
                model_path = self.pretrained_model_dir / f"{name}.keras"
                if model_path.exists():
                    self.models[name] = load_model(model_path)
                    logging.info(f"Loaded existing model {name} from {model_path}")

    def _prepare_input_data(self, X):
        if self.reshape_required and len(X.shape) == 2:
            return X.reshape((X.shape[0], 1, X.shape[1]))
        return X

    def save_training_history(self, history, model_name):
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f"{self.model_dir}/{model_name}_history.csv", index=False)

    def train_model(self, model, X_train, y_train, epochs=60, batch_size=32, **kwargs):  # Adjusted epochs and batch size
        X_train_prepared = self._prepare_input_data(np.asarray(X_train, dtype='float32'))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),  # Adjusted patience
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8)
        ]
        history = model.fit(
            X_train_prepared, np.asarray(y_train, dtype='float32'),
            validation_split=0.2,
            callbacks=callbacks,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )
        logging.info("Validation MAE: %s", history.history.get('val_mae', ['N/A'])[-1])
        self.save_training_history(history, model.name)
        return history

    def train_all_models(self, X_train, y_train, epochs=50, batch_size=32):  # Adjusted epochs and batch size
        results = {}
        try:
            for name, model in self.models.items():
                model_path = self.pretrained_model_dir / f"{name}.keras"
                if model_path.exists():
                    logging.info(f"Loading existing model: {name}")
                    self.models[name] = load_model(model_path)
                else:
                    logging.info(f"Training model: {name}")
                    self.train_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                    save_path = self.model_dir / f"{name}.keras"
                    if not save_path.exists():
                        model.save(save_path)
                        logging.info(f"Saved model {name} to {save_path}")
                results[name] = {"trained": True}
        except KeyboardInterrupt:
            logging.info("Training interrupted. Saving progress...")
            for name, model in self.models.items():
                save_path = self.model_dir / f"{name}.keras"
                if not save_path.exists():
                    model.save(save_path)
                    logging.info(f"Saved model {name} to {save_path}")
            raise
        return results

class ModelEvaluator:
    def __init__(self, feature_scaler=None, target_scaler=None):
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        Evaluate model performance ensuring correct data shapes and handling NaN values.
        """
        # Get predictions - expect X_test to be (samples, timesteps, features)
        predictions = model.predict(X_test)
        
        # Reshape predictions and y_test to 1D arrays
        predictions = predictions.reshape(-1)
        y_test_flat = y_test.reshape(-1)[:predictions.shape[0]]
        
        # Remove NaN values from both arrays
        mask = ~(np.isnan(predictions) | np.isnan(y_test_flat))
        predictions_clean = predictions[mask]
        y_test_clean = y_test_flat[mask]
        
        if len(predictions_clean) == 0:
            logging.warning("No valid predictions after removing NaN values")
            return {
                'mae': np.nan,
                'mse': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'test_loss': np.nan,
                'test_acc': np.nan,
                'n_samples': 0,
                'n_nans': np.sum(~mask)
            }
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_clean, predictions_clean)
        mse = mean_squared_error(y_test_clean, predictions_clean)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_clean, predictions_clean)
        
        # Model evaluation with cleaned data
        try:
            loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        except ValueError:
            loss, test_acc = -1, -1
            logging.warning("Could not calculate test accuracy due to shape mismatch")
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'test_loss': loss,
            'test_acc': test_acc,
            'n_samples': len(predictions_clean),
            'n_nans': np.sum(~mask)
        }

class HVACDataPreprocessor:
    def __init__(self, scaler_type: str = 'standard', imputer_n_neighbors: int = 5, country_holidays: str = 'US'):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors=imputer_n_neighbors)
        self.holidays = holidays.CountryHoliday(country_holidays)
        self.feature_names = None
        self.numerical_columns = None
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Fixed format string
        )
        self.logger = logging.getLogger(__name__)

    def _validate_raw_data(self, df: pd.DataFrame) -> None:
        required_columns = [
            'Date', 'on_off', 'damper', 'active_energy', 'co2_1', 'amb_humid_1',
            'active_power', 'pot_gen', 'high_pressure_1', 'high_pressure_2',
            'low_pressure_1', 'low_pressure_2', 'high_pressure_3', 'low_pressure_3',
            'outside_temp', 'outlet_temp', 'inlet_temp', 'summer_setpoint_temp',
            'winter_setpoint_temp', 'amb_temp_2'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        try:
            pd.to_datetime(df['Date'])
        except:
            raise ValueError("Date column cannot be parsed as datetime")

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        # First handle boolean columns
        boolean_columns = ['on_off', 'damper']
        df[boolean_columns] = df[boolean_columns].fillna(0)
        
        # Handle numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        
        # First try forward fill and backward fill for time series consistency
        df[numerical_columns] = df[numerical_columns].ffill().bfill()
        
        # If any NaNs remain, use KNN imputer
        if df[numerical_columns].isna().any().any():
            df[numerical_columns] = self.imputer.fit_transform(df[numerical_columns])
        
        # Verify no NaNs remain
        if df.isna().any().any():
            raise ValueError("Unable to handle all missing values in preprocessing")
        
        return df

    def _engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['datetime'] = pd.to_datetime(df['Date'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['is_holiday'] = df['datetime'].apply(lambda x: x in self.holidays).astype(int)
        return df

    def _engineer_hvac_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['temp_difference_in_out'] = df['outlet_temp'] - df['inlet_temp']
        df['temp_difference_ambient'] = df['outside_temp'] - df['inlet_temp']
        df['high_pressure_avg'] = df[['high_pressure_1', 'high_pressure_2', 'high_pressure_3']].mean(axis=1)
        df['low_pressure_avg'] = df[['low_pressure_1', 'low_pressure_2', 'low_pressure_3']].mean(axis=1)
        df['pressure_ratio'] = df['high_pressure_avg'] / (df['low_pressure_avg']+1e-6)
        df['power_per_temp_diff'] = df['active_power'] / (df['temp_difference_in_out'] + 1e-6)
        df['energy_efficiency'] = df['active_energy'] / (df['active_power'] + 1e-6)
        df['temp_setpoint_diff'] = np.where(df['month'].isin([6, 7, 8]),
                                             df['inlet_temp'] - df['summer_setpoint_temp'],
                                             df['inlet_temp'] - df['winter_setpoint_temp'])
        return df

    def _create_rolling_features(self, df: pd.DataFrame, windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """Create rolling features for key metrics"""
        key_metrics = ['active_power', 'inlet_temp', 'co2_1', 'amb_humid_1']
        
        for window in windows:
            for metric in key_metrics:
                df[f'{metric}_rolling_mean_{window}h'] = (
                    df[metric].rolling(window=window * 12, min_periods=1).mean()
                )
                df[f'{metric}_rolling_std_{window}h'] = (
                    df[metric].rolling(window=window * 12, min_periods=1).std()
                )
        return df

    def _prepare_target_variable(self, df: pd.DataFrame) -> tuple:
        target = df['active_power']
        features = df.drop(['active_power', 'datetime', 'Date'], axis=1)
        return features, target

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        return df

    def preprocess(self, df: pd.DataFrame, training: bool = True) -> tuple:
        self.logger.info("Starting preprocessing pipeline...")
        self._validate_raw_data(df)
        df = self._handle_missing_values(df)
        df = self._engineer_time_features(df)
        df = self._engineer_hvac_features(df)
        df = self._create_rolling_features(df)
        features, target = self._prepare_target_variable(df)
        features = self._scale_features(features)
        self.logger.info("Preprocessing pipeline completed successfully.")
        return features, target

    def get_feature_names(self) -> list:
        if self.feature_names is None:
            self.feature_names = list(self.scaler.feature_names_in_)
        return self.feature_names

class DataValidator:
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> dict:
        missing = df.isnull().sum().to_dict()
        return {'quality_score': 1.0, 'missing_columns': [k for k, v in missing.items() if v > 0]}  # Fixed items() call

class AnomalyDetector:
    def __init__(self, input_dim, method='autoencoder'):
        self.input_dim = input_dim
        self.method = method
        self.model = self._build_model()
        self.threshold = None
        self.scaler = StandardScaler()
        
    def _build_model(self):
        if self.method == 'autoencoder':
            return self._build_basic_autoencoder()
        elif self.method == 'complex_autoencoder':
            return self._build_complex_autoencoder()
        elif self.method == 'isolation_forest':
            return IsolationForest()
        else:
            raise ValueError("Unsupported method")

    def _build_basic_autoencoder(self):
        input_layer = Input(shape=(self.input_dim,))
        encoder = Dense(32, activation="relu")(input_layer)
        encoder = Dense(16, activation="relu")(encoder)
        encoder = Dropout(0.2)(encoder)
        decoder = Dense(32, activation="relu")(encoder)
        decoder = Dense(self.input_dim, activation="linear")(decoder)
        model = Model(inputs=input_layer, outputs=decoder)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _build_complex_autoencoder(self):
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = LayerNormalization()(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = LayerNormalization()(encoded)
        decoded = Dense(32, activation='relu')(encoded)
        decoded = LayerNormalization()(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = LayerNormalization()(decoded)
        decoded = Dense(self.input_dim, activation='linear')(decoded)
        model = Model(inputs=input_layer, outputs=decoded)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def fit(self, X, epochs=50, batch_size=32):
        X_scaled = self.scaler.fit_transform(X)
        if isinstance(self.model, Model):
            self.model.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
        else:
            self.model.fit(X_scaled)
        
    def detect_anomalies(self, X, threshold=None):
        X_scaled = self.scaler.transform(X)
        if isinstance(self.model, Model):
            errors = np.mean(np.square(X_scaled - self.model.predict(X_scaled)), axis=1)
            self.threshold = threshold or np.percentile(errors, 90)  # Adjusted threshold
            return errors > self.threshold
        else:
            return self.model.predict(X_scaled)

    def get_reconstruction(self, X):
        if not isinstance(self.model, Model):
            raise ValueError("Reconstruction only available for autoencoder models")
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled)
        return self.scaler.inverse_transform(reconstructed)
    
class HVACAnalysisPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.setup_directories()
        self.setup_logging()
        
    def setup_directories(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.config['output_base_dir']) / f"analysis_{timestamp}"
        self.subdirs = {
            'models': self.output_dir / 'models',
            'plots': self.output_dir / 'plots',
            'results': self.output_dir / 'results',
            'anomalies': self.output_dir / 'anomalies',
            'shap': self.output_dir / 'shap'
        }
        for dir_path in self.subdirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self) -> None:
        log_file = self.output_dir / 'pipeline.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Fixed format string
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> pd.DataFrame:
        self.logger.info(f"Loading data from {self.config['data_path']}")
        try:
            return pd.read_csv(self.config['data_path'], parse_dates=['Date'])
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
        
    def save_results(self, results: dict, filename: str) -> None:
        file_path = self.subdirs['results'] / filename
        def convert(o):
            return o.item() if hasattr(o, 'item') else o
        with open(file_path, 'w') as f:
            json.dump(results, f, default=convert)

    def plot_model_performance(self, results: dict) -> None:
        plt.figure(figsize=(12, 6))
        for model_name, history in results.get('training_history', {}).items():
            plt.plot(history.history.get('loss', []), label=model_name)
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.subdirs['plots'] / 'training_history.png')
        plt.close()
        plt.figure(figsize=(10, 6))
        metrics = pd.DataFrame(results.get('evaluation_results', {})).T
        if 'r2' in metrics.columns:
            metrics['r2'].plot(kind='bar')
        plt.title('Model R² Scores')
        plt.tight_layout()
        plt.savefig(self.subdirs['plots'] / 'model_comparison.png')
        plt.close()
        
    def run_pipeline(self) -> dict:
        try:
            # Load and preprocess data
            df = self.load_data()
            preprocessor = HVACDataPreprocessor(scaler_type='standard', imputer_n_neighbors=5)
            features, target = preprocessor.preprocess(df, training=True)
            
            # Prepare sequences for training
            sequence_length = self.config['model_params']['sequence_length']
            forecast_horizon = self.config['model_params']['forecast_horizon']
            X_sequences, y_sequences = preprocess_data(
                features.values,  # Convert DataFrame to numpy array
                target.values,    # Convert Series to numpy array
                sequence_length=sequence_length,
                forecast_horizon=forecast_horizon
            )
            
            # Split data into train/test sets
            split_idx = int(len(X_sequences) * 0.8)
            X_train = X_sequences[:split_idx]
            X_test = X_sequences[split_idx:]
            y_train = y_sequences[:split_idx]
            y_test = y_sequences[:split_idx:]
            
            # Initialize and train models
            trainer = ModelTrainer(
                input_shape=(sequence_length, features.shape[1]),
                output_dim=forecast_horizon,
                model_dir=self.subdirs['models']
            )
            trainer.load_existing_models()
            trainer.compile_models()
            training_results = trainer.train_all_models(X_train, y_train)
            
            # Evaluate models
            evaluator = ModelEvaluator(preprocessor.scaler)
            evaluation_results = {}
            for name, model in trainer.models.items():
                metrics = evaluator.evaluate_model(model, X_test, y_test)
                evaluation_results[name] = metrics
                self.logger.info(f"Model {name} evaluation results: {metrics}")
                
            results = {
                'training_history': training_results,
                'evaluation_results': evaluation_results
            }
            # Save each trained model
            for name, model in trainer.models.items():
                model_path = self.subdirs['models'] / f"{name}.keras"
                model.save(model_path)
                logging.info(f"Saved model {name} to {model_path}")
            # Run anomaly detection using autoencoder
            autoencoder = ModelBuilder.build_autoencoder(features.shape[1])
            # Prepare training data for autoencoder (flatten sequences)
            X_train_flat = X_train.reshape(-1, features.shape[1])
            autoencoder.fit(X_train_flat, X_train_flat, epochs=50, batch_size=32, verbose=0)
            # Detect anomalies on test data (flatten sequences)
            X_test_flat = X_test.reshape(-1, features.shape[1])
            reconstructions = autoencoder.predict(X_test_flat)
            reconstruction_errors = np.mean(np.square(X_test_flat - reconstructions), axis=1)
            anomaly_threshold = np.percentile(reconstruction_errors, 90)
            anomalies = (reconstruction_errors > anomaly_threshold).tolist()
            results['anomaly_detection'] = {
                'threshold': float(anomaly_threshold),
                'n_anomalies': int(sum(anomalies)),
                'reconstruction_errors': reconstruction_errors.tolist()  # Added reconstruction_errors to results
            }
            # Run SHAP analysis (for example, on best model using ExplainableAI)
            best_model_name = max(evaluation_results, key=lambda k: evaluation_results[k].get('r2', 0))
            explainer = ExplainableAI(output_dir=str(self.subdirs['shap']))
            best_model = trainer.models[best_model_name]
            # Use a sample from X_train_flat for background; adjust reshape as needed
            explainer.create_explainer(best_model, X_train_flat, model_type='lstm')
            shap_values = explainer.compute_shap_values(X_test_flat, model_type='lstm')
            explainer.plot_shap_summary('lstm', preprocessor.get_feature_names(), X_test_flat)
            results['shap'] = {'model': best_model_name, 'shap_summary': f"shap_summary_lstm.png"}
            return results
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            raise

    def generate_recommendations(self, results: dict) -> dict:
        recommendations = {
            'model_selection': {},
            'anomaly_detection': {},
            'system_optimization': {}
        }
        best_model = max(results.get('evaluation_results', {}), key=lambda k: results['evaluation_results'][k].get('r2', 0))
        recommendations['model_selection'] = {
            'best_model': best_model,
            'performance_metrics': results['evaluation_results'][best_model],
            'reason': f"Selected based on highest R² score of {results['evaluation_results'][best_model]['r2']:.3f}"
        }
        return recommendations

def preprocess_data(X, y, sequence_length=24, forecast_horizon=12):
    """
    Preprocess time series data into sequences for training.
    
    Args:
        X: Input features array (2D: samples, features)
        y: Target values array (1D)
        sequence_length: Length of input sequences
        forecast_horizon: Number of future steps to predict
    
    Returns:
        Tuple of (X sequences, y sequences)
    """
    # Ensure y is 1D
    if isinstance(y, pd.Series):
        y = y.values
    if len(y.shape) > 1:
        y = y.ravel()
    
    # Handle NaN values in input data
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        logging.warning(f"Found {np.sum(np.isnan(X))} NaN values in features and {np.sum(np.isnan(y))} in target")
        # Fill NaN values with forward fill, then backward fill
        X = pd.DataFrame(X).ffill().bfill().values
        y = pd.Series(y).ffill().bfill().values
    
    # Calculate valid number of sequences
    num_samples = len(y) - sequence_length - forecast_horizon + 1
    
    if num_samples <= 0:
        raise ValueError("Not enough samples to create sequences")
    
    # Initialize arrays with correct shapes
    num_features = X.shape[-1]
    X_sequences = np.zeros((num_samples, sequence_length, num_features))
    y_sequences = np.zeros((num_samples, forecast_horizon))
    
    # Create sequences without reshaping input
    for i in range(num_samples):
        X_sequences[i] = X[i:i + sequence_length]
        y_sequences[i] = y[i + sequence_length:i + sequence_length + forecast_horizon]
    
    return X_sequences, y_sequences

def process_hvac_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process HVAC data from raw file to model-ready format.
    
    Args:
        data_path: Path to raw data file
    Returns:
        Tuple of (processed features, target variable)
    """
    # Read data
    df = pd.read_csv(data_path, parse_dates=['Date'])
    
    # Initialize preprocessor
    preprocessor = HVACDataPreprocessor(
        scaler_type='standard',
        imputer_n_neighbors=5
    )
    
    # Validate data quality
    quality_report = DataValidator.check_data_quality(df)
    logging.info(f"Data quality report: {quality_report}")
    
    # Preprocess data
    features, target = preprocessor.preprocess(df, training=True)
    
    # Convert target to 1D array if needed
    if isinstance(target, pd.Series):
        target = target.values
    if len(target.shape) > 1:
        target = target.ravel()
    
    # Convert features to numpy array if it's a DataFrame
    if isinstance(features, pd.DataFrame):
        features = features.values
        
    return features, target

def train_and_evaluate_models(X_train, X_test, y_train, y_test, features, feature_scaler, target_scaler):
    # Initialize evaluator
    evaluator = ModelEvaluator(feature_scaler, target_scaler)
    
    # Train models
    trainer = ModelTrainer((1, len(features)))
    trainer.compile_models()
    training_results = trainer.train_all_models(X_train, y_train)
    
    # Evaluate models
    evaluation_results = {}
    for name, model in trainer.models.items():
        metrics = evaluator.evaluate_model(model, X_test, y_test)
        evaluation_results[name] = metrics
    
    # Find optimal settings for best model
    best_model_name = max(evaluation_results, key=lambda k: evaluation_results[k]['r2'])
    best_model = trainer.models[best_model_name]
    
    # Build autoencoder for anomaly detection
    autoencoder = ModelBuilder.build_autoencoder(len(features))
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)
    
    # Get optimal settings
    optimization_results = {}  # placeholder
    
    return {
        'models': trainer.models,
        'evaluation_results': evaluation_results,
        'optimization_results': optimization_results,
        'best_model_name': best_model_name
    }

class EnhancedHVACPipeline(HVACAnalysisPipeline):
    def __init__(self, config: dict):
        super().__init__(config)
        self.power_cost_per_kwh = config.get('power_cost_per_kwh', 0.12)  # Default electricity cost
        self.setup_ml_models()
        
    def setup_ml_models(self):
        """Initialize additional ML models for comparison"""
        self.ml_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=4,
                random_state=42
            )
        }

    def determine_comfort_range(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Determine comfort range from the data"""
        winter_temps = df[df['month'].isin([12, 1, 2])]['inlet_temp']
        summer_temps = df[df['month'].isin([6, 7, 8])]['inlet_temp']
        winter_comfort_range = (winter_temps.quantile(0.25), winter_temps.quantile(0.75))
        summer_comfort_range = (summer_temps.quantile(0.25), summer_temps.quantile(0.75))
        return winter_comfort_range, summer_comfort_range

    def optimize_setpoint(self, features: pd.DataFrame, current_temp: float, comfort_range: Tuple[float, float]) -> dict:
        """Optimize HVAC setpoint for minimum power consumption"""
        def power_consumption(setpoint):
            # Create feature vector with new setpoint
            test_features = features.copy()
            test_features['temp_setpoint'] = setpoint
            
            # Predict power consumption
            power = self.best_model.predict(test_features.values.reshape(1, -1))[0]
            
            # Penalty for comfort violation
            comfort_penalty = 0
            if setpoint < comfort_range[0]:
                comfort_penalty = (comfort_range[0] - setpoint) * 1000
            elif setpoint > comfort_range[1]:
                comfort_penalty = (setpoint - comfort_range[1]) * 1000
                
            return power + comfort_penalty

        # Optimize setpoint
        result = minimize(
            power_consumption,
            x0=current_temp,
            bounds=[(comfort_range[0]-2, comfort_range[1]+2)],
            method='Powell'
        )
        
        optimal_setpoint = result.x[0]
        predicted_power = power_consumption(optimal_setpoint) - comfort_penalty
        cost_savings = (power_consumption(current_temp) - predicted_power) * self.power_cost_per_kwh * 24
        
        return {
            'optimal_setpoint': optimal_setpoint,
            'predicted_power_savings': power_consumption(current_temp) - predicted_power,
            'daily_cost_savings': cost_savings,
            'current_power': power_consumption(current_temp)
        }

    def create_dashboard(self, df: pd.DataFrame, results: dict) -> None:
        """Create comprehensive dashboard using plotly"""
        # Create dashboard layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Power Consumption Pattern',
                'Temperature vs Power',
                'Model Performance Comparison',
                'Anomaly Detection',
                'SHAP Feature Importance',
                'Cost Analysis'
            )
        )
        
        # Power consumption pattern
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['active_power'],
                name='Power Consumption'
            ),
            row=1, col=1
        )
        
        # Temperature vs Power
        fig.add_trace(
            go.Scatter(
                x=df['inlet_temp'],
                y=df['active_power'],
                mode='markers',
                name='Temp vs Power'
            ),
            row=1, col=2
        )
        
        # Model performance comparison
        model_names = list(results['evaluation_results'].keys())
        r2_scores = [results['evaluation_results'][model]['r2'] for model in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=r2_scores, name='R² Score'),
            row=2, col=1
        )
        
        # Anomaly detection
        anomaly_scores = results['anomaly_detection']['reconstruction_errors']
        fig.add_trace(
            go.Scatter(
                x=df['datetime'][-len(anomaly_scores):],
                y=anomaly_scores,
                name='Anomaly Score'
            ),
            row=2, col=2
        )
        
        # SHAP importance
        feature_importance = results['shap']['feature_importance']
        fig.add_trace(
            go.Bar(
                x=feature_importance['feature'],
                y=feature_importance['importance'],
                name='SHAP Importance'
            ),
            row=3, col=1
        )
        
        # Cost analysis
        daily_costs = df.groupby(df['datetime'].dt.date)['active_power'].sum() * self.power_cost_per_kwh
        fig.add_trace(
            go.Scatter(
                x=daily_costs.index,
                y=daily_costs.values,
                name='Daily Cost'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(height=1200, width=1600, showlegend=True)
        
        # Save dashboard
        fig.write_html(str(self.subdirs['plots'] / 'dashboard.html'))

    def run_enhanced_pipeline(self) -> dict:
        """Run enhanced pipeline with all new features"""
        try:
            # Run basic pipeline first
            base_results = self.run_pipeline()
            
            # Load and preprocess data
            df = self.load_data()
            preprocessor = HVACDataPreprocessor()
            features, target = preprocessor.preprocess(df)
            
            # Determine comfort range from data
            winter_comfort_range, summer_comfort_range = self.determine_comfort_range(df)
            
            # Train ML models for comparison
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            ml_results = {}
            for name, model in self.ml_models.items():
                model.fit(X_train, y_train)
                ml_results[name] = {
                    'r2': r2_score(y_test, model.predict(X_test)),
                    'mae': mean_absolute_error(y_test, model.predict(X_test))
                }
            
            # Optimize setpoints
            current_conditions = features.iloc[-1]
            current_month = current_conditions['month']
            comfort_range = summer_comfort_range if current_month in [6, 7, 8] else winter_comfort_range
            optimization_results = self.optimize_setpoint(
                current_conditions,
                current_conditions['inlet_temp'],
                comfort_range
            )
            
            # Analyze usage patterns
            usage_patterns = self.analyze_usage_patterns(df)
            
            # Enhanced anomaly detection
            anomaly_detector = AnomalyDetector(features.shape[1], method='complex_autoencoder')
            anomaly_detector.fit(features.values)
            anomalies = anomaly_detector.detect_anomalies(features.values)
            
            # Create detailed SHAP analysis
            explainer = ExplainableAI(str(self.subdirs['shap']))
            best_model = self.ml_models[max(ml_results, key=lambda k: ml_results[k]['r2'])]
            explainer.create_explainer(best_model, X_train, 'ml')
            shap_values = explainer.compute_shap_values(X_test, 'ml')
            feature_importance = explainer.generate_feature_importance('ml', features.columns)
            
            # Combine all results
            enhanced_results = {
                **base_results,
                'ml_results': ml_results,
                'optimization_results': optimization_results,
                'usage_patterns': usage_patterns,
                'anomalies': {
                    'indices': np.where(anomalies)[0].tolist(),
                    'total_count': sum(anomalies),
                    'anomaly_dates': df.iloc[np.where(anomalies)[0]]['datetime'].tolist()
                },
                'shap': {
                    'feature_importance': feature_importance.to_dict(),
                    'summary_plot': 'shap_summary_ml.png'
                }
            }
            
            # Create dashboard
            self.create_dashboard(df, enhanced_results)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Enhanced pipeline error: {e}")
            raise

    def generate_recommendations(self, results: dict) -> dict:
        """Generate enhanced recommendations including cost and comfort optimization"""
        recommendations = super().generate_recommendations(results)
        
        # Add optimization recommendations
        recommendations['setpoint_optimization'] = {
            'optimal_setpoint': results['optimization_results']['optimal_setpoint'],
            'potential_savings': results['optimization_results']['daily_cost_savings'],
            'comfort_impact': 'Minimal - Within preferred range'
        }
        
        # Add usage pattern recommendations
        recommendations['usage_optimization'] = {
            'peak_hours': f"Consider reducing usage during {results['usage_patterns']['peak_hours']}",
            'off_peak_suggestion': f"Shift non-essential cooling to {results['usage_patterns']['off_peak_hours']}"
        }
        
        # Add anomaly-based recommendations
        if results['anomalies']['total_count'] > 0:
            recommendations['maintenance'] = {
                'anomaly_detected': True,
                'suggestion': 'Schedule maintenance check - Unusual patterns detected',
                'dates_of_concern': results['anomalies']['anomaly_dates'][-5:]  # Last 5 anomalies
            }
            
        return recommendations

if __name__ == "__main__":
    config = {
        'data_path': 'HVAC_dataset.csv',
        'output_base_dir': 'outputs',
        'power_cost_per_kwh': 0.15,  # Adjust based on local rates
        'model_params': {
            'sequence_length': 24,
            'forecast_horizon': 12,
            'batch_size': 32,
            'epochs': 100             # Increased epochs for better training
        }
    }
    
    pipeline = EnhancedHVACPipeline(config)
    results = pipeline.run_enhanced_pipeline()
    recommendations = pipeline.generate_recommendations(results)
    
    # Save all results
    pipeline.save_results(results, 'enhanced_results.json')
    pipeline.save_results(recommendations, 'enhanced_recommendations.json')
    
    # Log key findings
    logging.info(f"Optimal setpoint: {results['optimization_results']['optimal_setpoint']:.1f}°C")
    logging.info(f"Potential daily savings: ${results['optimization_results']['daily_cost_savings']:.2f}")
    logging.info(f"Number of anomalies detected: {results['anomalies']['total_count']}")


