import numpy as np
import pandas as pd
import logging
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Flatten
from tensorflow.keras.layers import Bidirectional, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelBuilder:
    @staticmethod
    def build_basic_lstm(input_shape, output_dim=12):  # Add output_dim parameter
        input_layer = Input(shape=input_shape)
        lstm_1 = LSTM(64, return_sequences=True)(input_layer)
        lstm_2 = LSTM(32, activation='relu')(lstm_1)
        output_layer = Dense(output_dim)(lstm_2)  # Match output dimension with target
        model = Model(inputs=input_layer, outputs=output_layer)
        return model

    @staticmethod
    def build_advanced_lstm(input_shape, output_dim=12):  # Add output_dim parameter
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(output_dim)  # Match output dimension with target
        ])
        return model

    @staticmethod
    def build_bi_lstm(input_shape, output_dim=12):  # Add output_dim parameter
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
            Bidirectional(LSTM(64, activation='relu')),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_dim)  # Match output dimension with target
        ])
        return model

    @staticmethod
    def build_attention(input_shape, output_dim=12):  # Add output_dim parameter
        input_layer = Input(shape=input_shape)
        attention_layer = MultiHeadAttention(num_heads=4, key_dim=8)(input_layer, input_layer)
        flatten_layer = Flatten()(attention_layer)
        dense_layer = Dense(64, activation='relu')(flatten_layer)
        output_layer = Dense(output_dim)(dense_layer)  # Match output dimension with target
        return Model(inputs=input_layer, outputs=output_layer)

    @staticmethod
    def build_combined_lstm_attention(input_shape, output_dim=12):  # Add output_dim parameter
        input_layer = Input(shape=input_shape)
        lstm_layer = LSTM(64, return_sequences=True)(input_layer)
        attention_layer = MultiHeadAttention(num_heads=4, key_dim=8)(lstm_layer, lstm_layer)
        attention_normalized = LayerNormalization()(attention_layer)
        dense_layer = Dense(64, activation='relu')(attention_normalized)
        dropout_layer = Dropout(0.2)(dense_layer)
        output_layer = Dense(output_dim)(dropout_layer)  # Match output dimension with target
        return Model(inputs=input_layer, outputs=output_layer)

    @staticmethod
    def build_model(input_shape):
        model = Sequential()
        # ...existing code...
        model.add(Dense(24 * 12, activation='linear'))  # Adjust the output layer
        model.add(layers.Reshape((24, 12)))  # Reshape to match the expected output shape
        # ...existing code...
        return model

class ModelTrainer:
    def __init__(self, input_shape, output_dim=12):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.models = {
            'bi_lstm': ModelBuilder.build_bi_lstm(self.input_shape, self.output_dim),
            'attention': ModelBuilder.build_attention(self.input_shape, self.output_dim),
            'combined': ModelBuilder.build_combined_lstm_attention(self.input_shape, self.output_dim)
        }

    def compile_models(self, learning_rate=0.001):
        for name, model in self.models.items():
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    def train_model(self, model, X_train, y_train, **kwargs):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        
        return model.fit(
            X_train, y_train,
            validation_split=0.2,
            callbacks=callbacks,
            **kwargs
        )

    def train_all_models(self, X_train, y_train, epochs=50, batch_size=32):
        results = {}
        for name, model in self.models.items():
            logging.info(f"Training {name} model...")
            history = self.train_model(
                model, X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            results[name] = history
        return results

class ModelEvaluator:
    def __init__(self, feature_scaler=None, target_scaler=None):
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1)
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        
        return {
            'mae': mean_absolute_error(y_test, predictions),
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions)
        }

    @staticmethod
    def cross_validate(models, X, y, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {}
        
        for name, model in models.items():
            metrics = {'mae': [], 'rmse': [], 'r2': []}
            
            for train_idx, val_idx in tscv.split(X):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                # Ensure proper data shape
                if len(X_train_cv.shape) == 2:
                    X_train_cv = X_train_cv.reshape((X_train_cv.shape[0], 1, X_train_cv.shape[1]))
                    X_val_cv = X_val_cv.reshape((X_val_cv.shape[0], 1, X_val_cv.shape[1]))
                
                # Train and evaluate
                early_stopping = EarlyStopping(monitor='val_loss', patience=5)
                model.fit(X_train_cv, y_train_cv, epochs=50, batch_size=32,
                         validation_split=0.2, callbacks=[early_stopping], verbose=0)
                
                # Calculate metrics
                val_metrics = ModelEvaluator.evaluate_model(model, X_val_cv, y_val_cv)
                metrics['mae'].append(val_metrics['mae'])
                metrics['rmse'].append(val_metrics['rmse'])
                metrics['r2'].append(val_metrics['r2'])
            
            # Calculate average metrics
            cv_results[name] = {
                'mae_mean': np.mean(metrics['mae']),
                'mae_std': np.std(metrics['mae']),
                'rmse_mean': np.mean(metrics['rmse']),
                'rmse_std': np.std(metrics['rmse']),
                'r2_mean': np.mean(metrics['r2']),
                'r2_std': np.std(metrics['r2'])
            }
            
        return cv_results

    def find_optimal_settings(self, model, autoencoder, features, feature_columns):
        """
        Find optimal temperature and current settings for minimum power consumption
        while maintaining comfort conditions.
        
        Returns:
        - Dictionary containing:
            - optimal_settings: DataFrame row with optimal parameters
            - min_power: Minimum predicted power consumption
            - optimal_temp: Optimal temperature
            - optimization_space: DataFrame with all evaluated combinations
        """
        # Define search spaces
        temperature_range = np.arange(18, 30, 1)
        current_range = np.arange(5, 15, 0.5)
        temp_grid, curr_grid = np.meshgrid(temperature_range, current_range)
        
        # Create parameter combinations
        settings_df = pd.DataFrame({
            'Inside Temperature (°C)': temp_grid.flatten(),
            'Current (A)': curr_grid.flatten()
        })
        
        # Add constant parameters
        settings_df['Voltage (V)'] = 220.0
        settings_df['Power Factor'] = 0.9
        settings_df['Outside Temperature (°C)'] = 25.0
        settings_df['Inside Humidity (%)'] = 50.0
        settings_df['Outside Humidity (%)'] = 50.0
        
        # Calculate derived features
        settings_df['Temperature Difference'] = (
            settings_df['Outside Temperature (°C)'] - 
            settings_df['Inside Temperature (°C)']
        )
        settings_df['Power Factor * Current'] = (
            settings_df['Power Factor'] * settings_df['Current (A)']
        )
        settings_df['Current Squared'] = settings_df['Current (A)'] ** 2
        
        # Normalize features
        if self.feature_scaler:
            normalized_data = self.feature_scaler.transform(settings_df[feature_columns])
            normalized_df = pd.DataFrame(normalized_data, columns=feature_columns)
        else:
            normalized_df = settings_df[feature_columns]
        
        # Prepare input for prediction
        model_input = normalized_df[features].values.reshape((-1, 1, len(features)))
        
        # Get predictions
        predictions = model.predict(model_input)
        if self.target_scaler:
            predictions = self.target_scaler.inverse_transform(predictions)
        
        # Get anomaly scores if autoencoder is provided
        if autoencoder is not None:
            reconstruction = autoencoder.predict(normalized_df[features].values)
            reconstruction_error = np.mean(np.power(
                normalized_df[features].values - reconstruction, 2
            ), axis=1)
            settings_df['anomaly_score'] = reconstruction_error
        
        # Add predictions to results
        settings_df['predicted_power'] = predictions.flatten()
        
        # Apply comfort constraints
        valid_mask = (
            (settings_df['Inside Temperature (°C)'] >= 21) & 
            (settings_df['Inside Temperature (°C)'] <= 25) & 
            (settings_df['Inside Humidity (%)'] >= 40) & 
            (settings_df['Inside Humidity (%)'] <= 60)
        )
        
        if autoencoder is not None:
            threshold = np.percentile(settings_df['anomaly_score'], 95)
            valid_mask &= (settings_df['anomaly_score'] < threshold)
        
        # Find optimal settings
        valid_settings = settings_df[valid_mask]
        if len(valid_settings) > 0:
            optimal_idx = valid_settings['predicted_power'].idxmin()
            optimal_settings = valid_settings.loc[optimal_idx]
            min_power = optimal_settings['predicted_power']
            optimal_temp = optimal_settings['Inside Temperature (°C)']
        else:
            optimal_settings = None
            min_power = None
            optimal_temp = None
        
        return {
            'optimal_settings': optimal_settings,
            'min_power': min_power,
            'optimal_temp': optimal_temp,
            'optimization_space': settings_df
        }

def perform_hyperparameter_tuning(X_train, y_train, input_shape):
    """Single implementation of hyperparameter tuning"""
    
    class CustomLSTMRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, neurons=64, learning_rate=0.001, epochs=50, batch_size=32):
            self.neurons = neurons
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.model = None

        def fit(self, X, y):
            self.model = ModelBuilder.build_basic_lstm(input_shape)
            self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
            self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
                verbose=0
            )
            return self

        def predict(self, X):
            return self.model.predict(X).reshape(-1)

    param_grid = {
        'neurons': [32, 64, 128],
        'learning_rate': [0.01, 0.001, 0.0001],
        'batch_size': [16, 32, 64]
    }

    grid = GridSearchCV(
        estimator=CustomLSTMRegressor(),
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_squared_error',
        n_jobs=1
    )

    grid_result = grid.fit(X_train, y_train)
    return grid_result.best_params_, grid_result.best_score_

# Example usage:
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
    optimization_results = evaluator.find_optimal_settings(
        best_model, 
        autoencoder,
        features,
        feature_columns
    )
    
    return {
        'models': trainer.models,
        'evaluation_results': evaluation_results,
        'optimization_results': optimization_results,
        'best_model_name': best_model_name
    }

# Remove all duplicate implementations of:
# - hyperparameter_tuning
# - predict_optimal_settings
# - evaluate_model functions