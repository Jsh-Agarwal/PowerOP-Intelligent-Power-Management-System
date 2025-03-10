import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from typing import Tuple, List, Dict

class Autoencoder:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.encoder, self.decoder, self.model = self._build_model()
        self.threshold = None
    
    def _build_model(self) -> Tuple[Model, Model, Model]:
        """Build autoencoder architecture."""
        # Encoder
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(32, activation='relu')(input_layer)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = Dense(8, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(16, activation='relu')(encoded)
        decoded = Dense(32, activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)
        
        # Models
        encoder = Model(input_layer, encoded)
        autoencoder = Model(input_layer, decoded)
        
        # Compile
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return encoder, Model(input_layer, decoded), autoencoder
    
    def train(
        self,
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict:
        """Train the autoencoder."""
        history = self.model.fit(
            data,
            data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Calculate reconstruction error threshold
        reconstructions = self.model.predict(data)
        errors = np.mean(np.square(data - reconstructions), axis=1)
        self.threshold = np.percentile(errors, 95)  # 95th percentile
        
        return history.history
    
    def detect_anomalies(
        self,
        data: np.ndarray
    ) -> Tuple[List[bool], np.ndarray]:
        """Detect anomalies in data."""
        reconstructions = self.model.predict(data)
        errors = np.mean(np.square(data - reconstructions), axis=1)
        
        if self.threshold is None:
            self.threshold = np.percentile(errors, 95)
        
        return errors > self.threshold, errors
    
    def save_model(self, model_path: str):
        """Save the model."""
        self.model.save(model_path)
        
    def load_model(self, model_path: str):
        """Load the model."""
        self.model = load_model(model_path)
        # Recreate encoder and decoder from loaded model
        self.encoder = Model(
            self.model.input,
            self.model.layers[2].output
        )
        self.decoder = Model(
            self.model.input,
            self.model.output
        )
