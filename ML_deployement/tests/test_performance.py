import pytest
import asyncio
import time
from locust import HttpUser, task, between
import numpy as np

from models.lstm_model import LSTMModel
from services.astra_db_service import AstraDBService

@pytest.mark.performance
class TestModelPerformance:
    """Test model performance metrics."""
    
    def test_lstm_inference_speed(self, lstm_model):
        """Test LSTM model inference speed."""
        input_data = np.random.normal(0, 1, (100, 24, 5))
        
        start_time = time.time()
        predictions = lstm_model.predict(input_data)
        end_time = time.time()
        
        inference_time = (end_time - start_time) / len(input_data)
        assert inference_time < 0.1  # Maximum 100ms per prediction

    def test_autoencoder_throughput(self, autoencoder):
        """Test autoencoder processing throughput."""
        data = np.random.normal(0, 1, (1000, 10))
        
        start_time = time.time()
        anomalies, _ = autoencoder.detect_anomalies(data)
        end_time = time.time()
        
        throughput = len(data) / (end_time - start_time)
        assert throughput > 100  # Minimum 100 samples per second

@pytest.mark.performance
class TestDatabasePerformance:
    """Test database performance metrics."""
    
    async def test_write_throughput(self, astra_service):
        """Test database write throughput."""
        test_data = [
            {
                "temperature": np.random.normal(22, 2),
                "humidity": np.random.normal(50, 5),
                "timestamp": datetime.now()
            }
            for _ in range(1000)
        ]
        
        start_time = time.time()
        await asyncio.gather(*[
            astra_service.save_temperature_data(data)
            for data in test_data
        ])
        end_time = time.time()
        
        throughput = len(test_data) / (end_time - start_time)
        assert throughput > 50  # Minimum 50 writes per second

class LoadTest(HttpUser):
    """Load testing for API endpoints."""
    
    wait_time = between(1, 2)
    
    @task
    def test_temperature_prediction(self):
        """Load test temperature prediction endpoint."""
        self.client.post(
            "/api/temperature/predict",
            json={
                "device_id": "test_device",
                "zone_id": "test_zone",
                "timestamps": [
                    datetime.now().isoformat()
                    for _ in range(24)
                ],
                "features": {
                    "temperature": [22.0] * 24,
                    "humidity": [50.0] * 24
                }
            },
            headers={"Authorization": "Bearer test_token"}
        )

    @task
    def test_system_optimization(self):
        """Load test system optimization endpoint."""
        self.client.post(
            "/api/optimize/system",
            json={
                "system_id": "test_system",
                "target_metric": "energy_efficiency",
                "constraints": {"max_temperature": 25.0},
                "current_state": {"temperature": 23.5}
            },
            headers={"Authorization": "Bearer test_token"}
        )

@pytest.mark.security
class TestSecurity:
    """Security testing."""
    
    def test_authentication(self, client):
        """Test authentication requirements."""
        response = client.get("/api/temperature/history")
        assert response.status_code == 401
        
        response = client.get(
            "/api/temperature/history",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401

    def test_input_validation(self, client, auth_headers):
        """Test input validation and sanitization."""
        response = client.post(
            "/api/temperature/predict",
            json={
                "device_id": "'; DROP TABLE users; --",
                "zone_id": "<script>alert('xss')</script>"
            },
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
