import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from models.lstm_model import LSTMModel
from models.autoencoder import Autoencoder
from services.weather_service import WeatherService
from services.astra_db_service import AstraDBService
from services.groq_slm_service import GroqSLMService
from optimization.comfort_optimization import ComfortOptimizer
from optimization.energy_optimization import EnergyOptimizer
from real_time.real_time_processing import RealTimeProcessor

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def synthetic_hvac_data():
    """Generate synthetic HVAC data for testing."""
    return {
        "temperature": np.random.normal(22, 2, 24),
        "humidity": np.random.normal(50, 5, 24),
        "pressure": np.random.normal(1013, 5, 24),
        "power": np.random.normal(1000, 100, 24),
        "flow_rate": np.random.normal(100, 10, 24),
        "timestamp": datetime.now()
    }

@pytest.fixture
def mock_weather_service():
    """Create mock weather service."""
    service = Mock(spec=WeatherService)
    service.get_forecast.return_value = {
        "temperature": 22.0,
        "humidity": 50.0,
        "pressure": 1013.0
    }
    return service

@pytest.fixture
def mock_astra_service():
    """Create mock Astra DB service."""
    service = Mock(spec=AstraDBService)
    return service

@pytest.fixture
def mock_groq_service():
    """Create mock Groq service."""
    service = Mock(spec=GroqSLMService)
    service.generate_hvac_optimization.return_value = {
        "recommendations": [],
        "expected_savings": 10.0,
        "confidence_score": 0.85
    }
    return service

@pytest.fixture
def test_lstm_model():
    """Create LSTM model for testing."""
    model = LSTMModel(input_shape=(24, 5))
    yield model
    model.close()

@pytest.fixture
def test_autoencoder():
    """Create autoencoder for testing."""
    model = Autoencoder(input_dim=10)
    yield model
    model.close()

@pytest.fixture
def comfort_optimizer(
    mock_weather_service,
    mock_astra_service,
    mock_groq_service
):
    """Create comfort optimizer with mock services."""
    optimizer = ComfortOptimizer(
        weather_service=mock_weather_service,
        db_service=mock_astra_service,
        groq_service=mock_groq_service
    )
    yield optimizer
    optimizer.close()

@pytest.fixture
def energy_optimizer(
    mock_weather_service,
    mock_astra_service,
    mock_groq_service
):
    """Create energy optimizer with mock services."""
    optimizer = EnergyOptimizer(
        weather_service=mock_weather_service,
        db_service=mock_astra_service,
        groq_service=mock_groq_service
    )
    yield optimizer
    optimizer.close()

@pytest.fixture
def real_time_processor(
    test_autoencoder,
    mock_astra_service
):
    """Create real-time processor for testing."""
    processor = RealTimeProcessor(
        autoencoder=test_autoencoder,
        db_service=mock_astra_service
    )
    yield processor
    processor.close()

@pytest.fixture
def mock_responses():
    """Common mock responses for testing."""
    return {
        "weather": {
            "success": {
                "temperature": 22.0,
                "humidity": 50.0,
                "pressure": 1013.0
            },
            "error": {"error": "API error", "code": 500}
        },
        "optimization": {
            "success": {
                "recommendations": [
                    {"action": "adjust_temperature", "value": 23.0}
                ],
                "expected_savings": 15.0
            },
            "error": {"error": "Optimization failed", "code": 500}
        }
    }

@pytest.fixture
def test_system_config():
    """Test system configuration."""
    return {
        "temperature_range": (18.0, 26.0),
        "humidity_range": (40.0, 60.0),
        "power_limit": 2000.0,
        "comfort_threshold": 80.0,
        "alert_thresholds": {
            "temperature": 2.0,
            "humidity": 10.0,
            "power": 500.0
        }
    }

@pytest.fixture
def performance_metrics():
    """Performance testing thresholds."""
    return {
        "prediction_latency": 0.1,  # seconds
        "optimization_latency": 0.5,  # seconds
        "throughput": 100,  # requests/second
        "memory_limit": 1024 * 1024 * 1024,  # 1GB
        "cpu_limit": 0.8  # 80% CPU usage
    }
