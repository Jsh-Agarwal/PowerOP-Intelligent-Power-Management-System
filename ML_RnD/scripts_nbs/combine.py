import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
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
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Flatten, Bidirectional, MultiHeadAttention, LayerNormalization, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import shap
import holidays

# Define all classes and functions from the individual files
from multiprocessing import Pool
import psutil
from tqdm import tqdm
import pickle  # Add this with other imports at the top

import lime
import lime.lime_tabular
from lime import submodular_pick
from IPython.display import HTML, display
import plotly.graph_objects as go
import plotly.express as px

class ExplainableAI:
    def __init__(self, output_dir: str = None):
        # Minimal initialization without file output
        self.logger = logging.getLogger('ExplainableAI')
        self.explainers = {}
        self.shap_values = {}
        self.feature_importance = {}
        self.feature_names = []
        self.lime_explainer = None
        self.class_names = ['normal', 'anomaly']
        
    def _setup_logger(self):
        logger = logging.getLogger('ExplainableAI')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.output_dir / 'xai.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def _prepare_background_data(self, X_train: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Prepare background data for SHAP analysis"""
        if len(X_train) > n_samples:
            indices = np.random.choice(len(X_train), n_samples, replace=False)
            background = X_train[indices]
        else:
            background = X_train
        return background

    def _reshape_data(self, data: np.ndarray, model_type: str) -> np.ndarray:
        """Reshape data based on model type"""
        if (model_type in ['lstm', 'bi_lstm', 'attention', 'combined'] and len(data.shape) == 2):
            return data.reshape(data.shape[0], 1, -1)
        return data

    def explain_predictions(self, model, X_train: np.ndarray, X_test: np.ndarray, 
                          feature_names: List[str], model_type: str) -> Dict[str, np.ndarray]:
        """Generate SHAP explanations with comprehensive error handling"""
        try:
            self.logger.info(f"Starting SHAP analysis for {model_type} model")
            
            # Prepare data
            background = self._prepare_background_data(X_train)
            background = self._reshape_data(background, model_type)
            X_test_sample = self._reshape_data(X_test[:100], model_type)  # Limit test samples for speed
            
            # Create explainer based on SHAP version
            try:
                # Try Explainer (newer SHAP versions)
                explainer = shap.Explainer(model, background)
                shap_values = explainer(X_test_sample)
                
                # Extract values from SHAP object
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values
                
                # Calculate feature importance
                feature_importance = np.abs(shap_values).mean(axis=0)
                if len(feature_importance.shape) > 1:
                    feature_importance = feature_importance.mean(axis=0)
                
                # Store results
                self.explainers[model_type] = explainer
                self.shap_values[model_type] = shap_values
                self.feature_importance[model_type] = feature_importance
                self.feature_names = feature_names
                
                # Generate visualizations
                self._generate_visualizations(shap_values, X_test_sample, feature_names, model_type)
                
                return {
                    'shap_values': shap_values,
                    'feature_importance': feature_importance,
                    'status': 'success'
                }
                
            except Exception as e:
                self.logger.error(f"Primary explainer failed: {str(e)}")
                # Fallback to GradientExplainer
                try:
                    explainer = shap.GradientExplainer(model, background)
                    shap_values = explainer.shap_values(X_test_sample)
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    
                    return self._process_kernel_explainer_results(shap_values, X_test_sample, feature_names, model_type)
                    
                except Exception as e2:
                    self.logger.error(f"Gradient explainer also failed: {str(e2)}")
                    raise
                
        except Exception as e:
            self.logger.error(f"SHAP analysis failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}

    def _process_kernel_explainer_results(self, shap_values, X_test_sample, feature_names, model_type):
        """Process results from KernelExplainer"""
        try:
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            self.shap_values[model_type] = shap_values
            self.feature_importance[model_type] = feature_importance
            
            self._generate_visualizations(shap_values, X_test_sample, feature_names, model_type)
            
            return {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'status': 'success'
            }
        except Exception as e:
            self.logger.error(f"Failed to process KernelExplainer results: {str(e)}")
            return {'status': 'failed', 'error': str(e)}

    def _generate_visualizations(self, shap_values: np.ndarray, X_test: np.ndarray, 
                               feature_names: List[str], model_type: str) -> None:
        # Skip saving visualizations
        pass

    def get_feature_importance(self, model_type: str) -> pd.DataFrame:
        """Get feature importance rankings"""
        if model_type not in self.feature_importance:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance[model_type]
        })
        return importance_df.sort_values('importance', ascending=False)

    def save_results(self, model_type: str) -> None:
        # Skip saving results
        pass
    
    def setup_lime_explainer(self, X_train):
        """Setup LIME explainer for local interpretability"""
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='regression'
        )

    def explain_instance(self, model, instance, num_features=10):
        """Generate LIME explanation for a single instance"""
        exp = self.lime_explainer.explain_instance(
            instance, 
            model.predict,
            num_features=num_features
        )
        return exp

    def generate_explanation_plots(self, shap_values, feature_names, model_type):
        """Generate comprehensive explanation plots"""
        try:
            # Feature importance plot using plotly
            importance_vals = np.abs(shap_values).mean(axis=0)
            fig = go.Figure([
                go.Bar(
                    x=importance_vals,
                    y=feature_names,
                    orientation='h'
                )
            ])
            fig.update_layout(
                title=f'Feature Importance ({model_type})',
                xaxis_title='SHAP value magnitude',
                yaxis_title='Features',
                height=800
            )
            fig.write_html(str(self.output_dir / f'feature_importance_{model_type}.html'))

            # SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'shap_summary_{model_type}.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to generate explanation plots: {e}")

class ModelBuilder:
    @staticmethod
    def build_basic_lstm(input_shape, output_dim=12):
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
    def build_autoencoder(input_dim, encoding_dim=16):
        inputs = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(inputs)
        decoded = Dense(input_dim, activation='linear')(encoded)
        autoencoder = Model(inputs=inputs, outputs=decoded)
        autoencoder.compile(optimizer=Adam(), loss='mse')
        return autoencoder

class ModelTrainer:
    def __init__(self, input_shape, output_dim=12, sequence_length=24):
        # Fix input shape handling
        if len(input_shape) == 2:
            self.input_shape = (sequence_length, input_shape[1])
        elif len(input_shape) == 3:
            self.input_shape = (input_shape[1], input_shape[2])
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")
        
        self.output_dim = output_dim
        self.models = {
            'bi_lstm': None,  # Initialize later
            'attention': None,
            'combined': None
        }
        self.sequence_length = sequence_length
        self.reshape_required = True
        self.model_dir = Path("D:/PowerAmp/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_models()  # New method

    def _initialize_models(self):
        """Initialize models with proper input shapes"""
        # Create attention layers first
        attention_layer1 = MultiHeadAttention(num_heads=2, key_dim=64)
        attention_layer2 = MultiHeadAttention(num_heads=2, key_dim=64)

        self.models = {
            'bi_lstm': Sequential([
                Input(shape=self.input_shape),
                Bidirectional(LSTM(128, return_sequences=True)),
                Bidirectional(LSTM(64)),
                Dense(32, activation='relu'),
                Dense(self.output_dim)
            ]),
            'attention': Sequential([
                Input(shape=self.input_shape),
                LSTM(128, return_sequences=True),
                # Use Lambda with pre-created attention layer
                Lambda(lambda x: attention_layer1(x, x, x, training=False)),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(self.output_dim)
            ]),
            'combined': Sequential([
                Input(shape=self.input_shape),
                Bidirectional(LSTM(128, return_sequences=True)),
                # Use Lambda with pre-created attention layer
                Lambda(lambda x: attention_layer2(x, x, x, training=False)),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(self.output_dim)
            ])
        }
        self.compile_models()

    def compile_models(self, learning_rate=0.0005):
        for name, model in self.models.items():
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']
            )

    def load_existing_models(self):
        """Load existing models from model_dir"""
        for name in self.models.keys():
            model_path = self.model_dir / f"{name}.keras"
            if model_path.exists():
                try:
                    self.models[name] = load_model(model_path)
                    logging.info(f"Loaded existing model {name} from {model_path}")
                except Exception as e:
                    logging.warning(f"Failed to load model {name}: {e}")
                    # Continue with new model initialization if loading fails
                    pass

    def _prepare_input_data(self, X):
        if self.reshape_required and len(X.shape) == 2:
            return X.reshape((X.shape[0], 1, X.shape[1]))
        return X

    def save_training_history(self, history, model_name):
        # Skip saving training history
        pass

    def train_model(self, model, X_train, y_train, epochs=60, batch_size=32, **kwargs):
        X_train_prepared = self._prepare_input_data(np.asarray(X_train, dtype='float32'))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
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

    def train_all_models(self, X_train, y_train, epochs=50, batch_size=32):
        results = {}
        for name, model in self.models.items():
            model_path = self.model_dir / f"{name}.keras"
            if model_path.exists():
                logging.info(f"Loading existing model: {name}")
                self.models[name] = load_model(model_path)
            else:
                logging.info(f"Training model: {name}")
                self.train_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            results[name] = {"trained": True}
        return results

class ModelEvaluator:
    def __init__(self, feature_scaler=None, target_scaler=None):
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """Evaluate model performance ensuring correct data shapes and handling NaN values."""
        # Get predictions - expect X_test to be (samples, timesteps, features)
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1)[:predictions.shape[0]]
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
        mae = mean_absolute_error(y_test_clean, predictions_clean)
        mse = mean_squared_error(y_test_clean, predictions_clean)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_clean, predictions_clean)
        try:
            loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        except ValueError:
            loss, test_acc = np.nan, np.nan
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
        return {'quality_score': 1.0, 'missing_columns': [k for k, v in missing.items() if v > 0]}

class AnomalyDetector:
    def __init__(self, input_dim, method='autoencoder'):
        self.input_dim = input_dim
        self.method = method
        self.model = self._build_model()
        self.threshold = None
        self.scaler = StandardScaler()
        self.verbose = True

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
            if self.verbose:
                print("Training autoencoder for anomaly detection...")
            # Add progress callback
            progbar = tf.keras.callbacks.Progbar(epochs)
            self.model.fit(
                X_scaled, X_scaled,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[progbar]
            )
        else:
            if self.verbose:
                print("Fitting isolation forest...")
            self.model.fit(X_scaled)

    def detect_anomalies(self, X, threshold=None):
        X_scaled = self.scaler.transform(X)
        if isinstance(self.model, Model):
            errors = np.mean(np.square(X_scaled - self.model.predict(X_scaled)), axis=1)
            self.threshold = threshold or np.percentile(errors, 90)
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
        self.model_dir = Path("D:/PowerAmp/models")  # Move this line before setup_directories()
        self.setup_directories()
        self.setup_logging()

    def setup_directories(self) -> None:
        # Simplified directory structure - only keep model directory
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up old files if they exist
        for file in self.model_dir.glob('*.keras'):
            file.unlink()
        for file in self.model_dir.glob('*.pkl'):
            file.unlink()

    def setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]  # Only log to console
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
        # Only save critical results like model performance metrics
        if filename == 'recommendations.json':
            file_path = self.model_dir / filename
            def convert(o):
                return o.item() if hasattr(o, 'item') else o
            with open(file_path, 'w') as f:
                json.dump(results, f, default=convert)

    def _save_model(self, model, name: str) -> None:
        """Save only the model file"""
        model_path = self.model_dir / f"{name}.keras"
        model.save(model_path)
        logging.info(f"Saved model {name} to {model_path}")

    def run_pipeline(self) -> dict:
        try:
            # Load and preprocess data Analysis Pipeline ===")
            df = self.load_data()
            preprocessor = HVACDataPreprocessor(scaler_type='standard', imputer_n_neighbors=5)
            
            # Fix date parsing
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
            
            features, target = preprocessor.preprocess(df, training=True)
            print(f"Loaded {len(df)} records")
            # Prepare sequences for training
            preprocessor = HVACDataPreprocessor(scaler_type='standard', imputer_n_neighbors=5)
            features, target = preprocessor.preprocess(df, training=True)
            print(f"Preprocessed {features.shape[1]} features")

            print("\n2. Preparing sequences...")
            sequence_length = self.config['model_params']['sequence_length']
            forecast_horizon = self.config['model_params']['forecast_horizon']
            X_sequences, y_sequences = preprocess_data(
                features.values,
                target.values,
                sequence_length=sequence_length,
                forecast_horizon=forecast_horizon
            )
            print(f"Created {len(X_sequences)} sequences")

            print("\n3. Splitting data...")
            split_idx = int(len(X_sequences) * 0.8)
            X_train = X_sequences[:split_idx]
            X_test = X_sequences[split_idx:]
            y_train = y_sequences[:split_idx]
            y_test = y_sequences[split_idx:]  # Fixed slicing
            print(f"Training set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")

            print("\n4. Initializing and training models...")
            trainer = ModelTrainer(
                input_shape=(sequence_length, features.shape[1]),
                output_dim=forecast_horizon
            )  # Remove model_dir argument since it's handled in ModelTrainer
            
            print("Loading or initializing models...")
            trainer.load_existing_models()  # This will now work correctly
            trainer.compile_models()
            training_results = trainer.train_all_models(X_train, y_train)
            print("Model training completed")

            print("\n5. Evaluating models...")
            evaluator = ModelEvaluator(preprocessor.scaler)
            evaluation_results = {}
            for name, model in trainer.models.items():
                print(f"\nEvaluating {name}...")
                metrics = evaluator.evaluate_model(model, X_test, y_test)
                evaluation_results[name] = metrics
                print(f"{name} R² score: {metrics['r2']:.4f}")

            results = {
                'training_history': training_results,
                'evaluation_results': evaluation_results
            }

            # Save only the best model
            best_model_name = max(evaluation_results, key=lambda k: evaluation_results[k].get('r2', 0))
            best_model = trainer.models[best_model_name]
            self._save_model(best_model, best_model_name)

            # Simplified results dictionary
            results = {
                'best_model': best_model_name,
                'metrics': evaluation_results[best_model_name]
            }

            return results

        except KeyboardInterrupt:
            self.logger.error("Pipeline interrupted by user.")
            raise
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
    """Preprocess time series data into sequences for training."""
    # Ensure y is 1D
    if isinstance(y, pd.Series):
        y = y.values
    if len(y.shape) > 1:
        y = y.ravel()
    
    # Handle NaN values in input data
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        logging.warning(f"Found {np.sum(np.isnan(X))} NaN values in features and {np.sum(np.isnan(y))} in target")
        X = pd.DataFrame(X).ffill().bfill().values
        y = pd.Series(y).ffill().bfill().values
    
    # Calculate valid number of sequences
    num_samples = len(y) - sequence_length - forecast_horizon + 1
    if num_samples <= 0:
        raise ValueError("Not enough samples to create sequences")
    
    # Create sequences
    num_features = X.shape[-1]
    X_sequences = np.zeros((num_samples, sequence_length, num_features))
    y_sequences = np.zeros((num_samples, forecast_horizon))
    
    for i in range(num_samples):
        X_sequences[i] = X[i:i + sequence_length]
        y_sequences[i] = y[i + sequence_length:i + sequence_length + forecast_horizon]
    
    return X_sequences, y_sequences

if __name__ == "__main__":
    import sys
    
    # First check if the dataset exists
    data_path = 'D:/PowerAmp/HVAC_dataset.csv'
    if not os.path.exists(data_path):
        print(f"\nError: HVAC dataset not found at {data_path}")
        print("Please ensure the dataset file exists at the specified location.")
        sys.exit(1)
    
    try:
        print("\nInitializing HVAC Analysis Pipeline...")
        config = {
            'output_base_dir': 'D:/PowerAmp/outputs',
            'data_path': 'D:/PowerAmp/HVAC_dataset.csv',
            'feature_names': [
                'Date', 'on_off', 'damper', 'active_energy', 'co2_1', 'amb_humid_1',
                'active_power', 'pot_gen', 'high_pressure_1', 'high_pressure_2',
                'low_pressure_1', 'low_pressure_2', 'high_pressure_3', 'low_pressure_3',
                'outside_temp', 'outlet_temp', 'inlet_temp', 'summer_setpoint_temp',
                'winter_setpoint_temp', 'amb_temp_2'
            ],
            'model_params': {
                'sequence_length': 24,
                'forecast_horizon': 12,
                'epochs': 50,
                'batch_size': 32
            }
        }
        
        pipeline = HVACAnalysisPipeline(config)
        results = pipeline.run_pipeline()
        
        print("\nSaving best model and metrics...")
        pipeline.save_results(results, 'recommendations.json')
        
        print("\nAnalysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\nError: Could not find required file: {str(e)}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise
