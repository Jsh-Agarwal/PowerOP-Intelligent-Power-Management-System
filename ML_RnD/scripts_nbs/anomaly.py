import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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
            return IsolationForest(contamination=0.1, random_state=42)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _build_basic_autoencoder(self):
        input_layer = Input(shape=(self.input_dim,))
        # Encoder
        encoder = Dense(32, activation="relu")(input_layer)
        encoder = Dense(16, activation="relu")(encoder)
        encoder = Dropout(0.2)(encoder)
        # Decoder
        decoder = Dense(32, activation="relu")(encoder)
        decoder = Dense(self.input_dim, activation="linear")(decoder)
        
        model = Model(inputs=input_layer, outputs=decoder)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _build_complex_autoencoder(self):
        input_layer = Input(shape=(self.input_dim,))
        # Enhanced encoder
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = LayerNormalization()(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = LayerNormalization()(encoded)
        
        # Enhanced decoder
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
        
        if isinstance(self.model, Model):  # Autoencoder models
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            history = self.model.fit(
                X_scaled, X_scaled,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Set threshold based on reconstruction error
            predictions = self.model.predict(X_scaled)
            reconstruction_errors = np.mean(np.power(X_scaled - predictions, 2), axis=1)
            self.threshold = np.percentile(reconstruction_errors, 95)
            
            return history
        else:  # Isolation Forest
            self.model.fit(X_scaled)
            return None

    def detect_anomalies(self, X, threshold=None):
        X_scaled = self.scaler.transform(X)
        
        if isinstance(self.model, Model):  # Autoencoder models
            predictions = self.model.predict(X_scaled)
            reconstruction_errors = np.mean(np.power(X_scaled - predictions, 2), axis=1)
            threshold = threshold or self.threshold
            return {
                'anomalies': reconstruction_errors > threshold,
                'scores': reconstruction_errors,
                'threshold': threshold
            }
        else:  # Isolation Forest
            predictions = self.model.predict(X_scaled)
            return {
                'anomalies': predictions == -1,
                'scores': self.model.score_samples(X_scaled),
                'threshold': None
            }

    def get_reconstruction(self, X):
        """Get reconstructed data for autoencoder models"""
        if not isinstance(self.model, Model):
            raise ValueError("Reconstruction only available for autoencoder models")
        
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled)
        return self.scaler.inverse_transform(reconstructed)

class AnomalyDetectorEnsemble:
    def __init__(self, input_dim):
        self.detectors = {
            'basic_autoencoder': AnomalyDetector(input_dim, 'autoencoder'),
            'complex_autoencoder': AnomalyDetector(input_dim, 'complex_autoencoder'),
            'isolation_forest': AnomalyDetector(input_dim, 'isolation_forest')
        }
        
    def fit(self, X, epochs=50, batch_size=32):
        results = {}
        for name, detector in self.detectors.items():
            results[name] = detector.fit(X, epochs, batch_size)
        return results
    
    def detect_anomalies(self, X):
        results = {}
        for name, detector in self.detectors.items():
            results[name] = detector.detect_anomalies(X)
        return results
    
    def get_ensemble_predictions(self, X, voting='majority'):
        all_predictions = self.detect_anomalies(X)
        
        if voting == 'majority':
            # Stack predictions from all detectors
            predictions = np.column_stack([
                pred['anomalies'] for pred in all_predictions.values()
            ])
            # Return True if majority of detectors flag as anomaly
            return np.mean(predictions, axis=1) > 0.5
        
        return all_predictions

# Example usage:
def analyze_anomalies(X_train, X_test):
    """
    Analyze anomalies using multiple methods and compare results.
    
    Returns:
    - Dictionary containing results from all methods and ensemble predictions
    """
    ensemble = AnomalyDetectorEnsemble(X_train.shape[1])
    
    # Train all detectors
    training_results = ensemble.fit(X_train)
    
    # Get predictions from all methods
    predictions = ensemble.detect_anomalies(X_test)
    
    # Get ensemble predictions
    ensemble_predictions = ensemble.get_ensemble_predictions(X_test)
    
    return {
        'individual_results': predictions,
        'ensemble_predictions': ensemble_predictions,
        'training_results': training_results
    }
