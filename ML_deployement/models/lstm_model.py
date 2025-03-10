import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import Dict, List, Tuple, Any
from datetime import datetime
from utils.exceptions import ModelError  # Add this import

class LSTMModel:
    def __init__(self, input_shape: Tuple[int, int]):
        self.input_shape = input_shape
        self.model = self._build_model()
        self.scaler = MinMaxScaler(feature_range=(18, 32))  # More realistic temperature range
        
        # Initialize with sensible ranges for HVAC data
        sample_data = np.array([
            [18, 30, 500],   # min values [temp, humidity, power]
            [32, 90, 3000]   # max values
        ])
        self.scaler.fit(sample_data)
        
    def _build_model(self) -> Sequential:
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(128, input_shape=self.input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def preprocess_data(
        self,
        data: Any,
        feature_columns: List[str],
        target_column: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for training."""
        # Scale features
        features = data[feature_columns].values
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        
        # Create sequences
        X, y = [], []
        sequence_length = self.input_shape[0]
        
        for i in range(len(scaled_features) - sequence_length):
            X.append(scaled_features[i:i + sequence_length])
            y.append(data[target_column].values[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict:
        """Train the model."""
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    async def predict_next_24h(
        self,
        current_data: Dict[str, List[float]],
        weather_forecast: Any = None
    ) -> Dict[str, List[float]]:
        """Predict next 24 hours."""
        try:
            # Ensure ordered features and proper shape
            features = np.array([
                current_data.get(feature, [22.0] * len(current_data['temperature']))
                for feature in ['temperature', 'humidity', 'power']
            ]).T
            
            # Validate input shape
            if features.shape[1] != self.input_shape[1]:
                raise ModelError(f"Expected {self.input_shape[1]} features, got {features.shape[1]}")
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            predictions = []
            
            # Generate predictions with initial sequence
            current_sequence = scaled_features[-self.input_shape[0]:].reshape((1, *self.input_shape))
            
            for _ in range(24):
                next_value = self.model.predict(current_sequence, verbose=0)
                # Scale prediction to temperature range
                temp_pred = np.clip(next_value[0, 0] * (32 - 18) + 18, 18, 32)
                predictions.append(temp_pred)
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1] = next_value[0, 0]
            
            # Calculate confidence based on prediction horizon
            confidences = [max(0.5, 1.0 - i * 0.02) for i in range(len(predictions))]
            
            return {
                "predictions": predictions,
                "confidence": confidences,
                "min_values": [p - c for p, c in zip(predictions, confidences)],
                "max_values": [p + c for p, c in zip(predictions, confidences)]
            }
            
        except Exception as e:
            raise ModelError(f"Prediction failed: {str(e)}")
    
    def save_model(self, model_path: str, scaler_path: str = None):
        """Save model and scaler."""
        self.model.save(model_path)
        if scaler_path:
            joblib.dump(self.scaler, scaler_path)
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """Load model and scaler."""
        self.model = load_model(model_path)
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
