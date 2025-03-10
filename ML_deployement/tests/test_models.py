import pytest
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf

from models.lstm_model import LSTMModel
from models.autoencoder import Autoencoder

@pytest.fixture
def lstm_model():
    """Create LSTM model fixture."""
    return LSTMModel(input_shape=(24, 5))

@pytest.fixture
def autoencoder():
    """Create autoencoder fixture."""
    return Autoencoder(input_dim=10)

@pytest.mark.models
class TestLSTMModel:
    """Test LSTM model functionality."""
    
    def test_model_initialization(self, lstm_model):
        """Test model initialization."""
        assert lstm_model.model is not None
        assert isinstance(lstm_model.scaler, MinMaxScaler)

    @pytest.mark.asyncio
    async def test_temperature_prediction(self, lstm_model):
        """Test temperature prediction."""
        # Generate synthetic data
        data = {
            'temperature': np.random.normal(22, 2, 48),
            'humidity': np.random.normal(50, 5, 48),
            'pressure': np.random.normal(1013, 5, 48),
            'power': np.random.normal(1000, 100, 48),
            'flow_rate': np.random.normal(100, 10, 48)
        }
        
        predictions = await lstm_model.predict_next_24h(
            pd.DataFrame(data),
            pd.DataFrame({'temperature': np.random.normal(22, 2, 24)})
        )
        
        assert len(predictions['predictions']) == 24
        assert all(15 <= temp <= 30 for temp in predictions['predictions'])

    def test_model_persistence(self, lstm_model, tmp_path):
        """Test model saving and loading."""
        model_path = tmp_path / "lstm_model.h5"
        scaler_path = tmp_path / "scaler.joblib"
        
        # Save model
        lstm_model.save_model(str(model_path), str(scaler_path))
        
        # Load model
        new_model = LSTMModel(input_shape=(24, 5))
        new_model.load_model(str(model_path), str(scaler_path))
        
        assert new_model.model.get_config() == lstm_model.model.get_config()

@pytest.mark.models
class TestAutoencoder:
    """Test autoencoder functionality."""
    
    def test_anomaly_detection(self, autoencoder):
        """Test anomaly detection."""
        # Generate normal data
        normal_data = np.random.normal(0, 1, (100, 10))
        
        # Generate anomalous data
        anomaly_data = np.random.normal(5, 1, (20, 10))
        
        # Train autoencoder
        autoencoder.train(normal_data, epochs=5)
        
        # Test detection
        anomalies, _ = autoencoder.detect_anomalies(np.vstack([normal_data, anomaly_data]))
        
        # Verify anomalies are detected
        assert sum(anomalies[100:]) > sum(anomalies[:100])

    def test_reconstruction_error(self, autoencoder):
        """Test reconstruction error calculation."""
        data = np.random.normal(0, 1, (50, 10))
        autoencoder.train(data, epochs=5)
        
        _, errors = autoencoder.detect_anomalies(data)
        assert len(errors) == len(data)
        assert all(error >= 0 for error in errors)

    @pytest.mark.asyncio
    async def test_real_time_processing(self, autoencoder):
        """Test real-time data processing."""
        data = pd.DataFrame({
            'metric1': np.random.normal(0, 1, 100),
            'metric2': np.random.normal(0, 1, 100)
        })
        
        results = await autoencoder.process_real_time_data(data)
        assert 'anomalies' in results
        assert 'reconstruction_errors' in results
        assert len(results['anomalies']) == len(data)
