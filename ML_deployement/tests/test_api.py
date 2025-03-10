import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

from main import app
from api.schemas import TemperaturePredictionRequest

@pytest.fixture
def client():
    """Create test client fixture."""
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Create authenticated headers fixture."""
    return {"Authorization": "Bearer test_token"}

@pytest.mark.api
class TestTemperatureEndpoints:
    """Test temperature-related endpoints."""
    
    def test_temperature_prediction(self, client, auth_headers):
        """Test temperature prediction endpoint."""
        request_data = {
            "device_id": "test_device",
            "zone_id": "test_zone",
            "timestamps": [
                (datetime.now() + timedelta(hours=i)).isoformat()
                for i in range(24)
            ],
            "features": {
                "temperature": [22.0] * 24,
                "humidity": [50.0] * 24
            }
        }
        
        response = client.post(
            "/api/temperature/predict",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "timestamps" in data
        assert len(data["predictions"]) == 24

    def test_temperature_history(self, client, auth_headers):
        """Test temperature history endpoint."""
        response = client.get(
            "/api/temperature/history",
            params={
                "device_id": "test_device",
                "zone_id": "test_zone",
                "start_time": (datetime.now() - timedelta(days=1)).isoformat()
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

@pytest.mark.api
class TestOptimizationEndpoints:
    """Test optimization-related endpoints."""
    
    def test_system_optimization(self, client, auth_headers):
        """Test system optimization endpoint."""
        request_data = {
            "system_id": "test_system",
            "target_metric": "energy_efficiency",
            "constraints": {
                "max_temperature": 25.0,
                "min_temperature": 20.0
            },
            "current_state": {
                "temperature": 23.5,
                "humidity": 55.0
            }
        }
        
        response = client.post(
            "/api/optimize/system",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "expected_savings" in data

@pytest.mark.api
class TestMonitoringEndpoints:
    """Test monitoring-related endpoints."""
    
    def test_system_status(self, client, auth_headers):
        """Test system status endpoint."""
        response = client.get(
            "/api/status/system/test_system",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "metrics" in data

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "services" in data
