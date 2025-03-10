import pytest
from unittest.mock import Mock, patch
import aiohttp
import numpy as np
from datetime import datetime

from services.weather_service import WeatherService
from services.astra_db_service import AstraDBService
from services.groq_slm_service import GroqSLMService

@pytest.fixture
async def weather_service():
    """Create weather service fixture."""
    service = WeatherService()
    yield service
    await service.close()

@pytest.fixture
def astra_service():
    """Create Astra DB service fixture."""
    service = AstraDBService()
    yield service
    service.close()

@pytest.fixture
async def groq_service():
    """Create Groq SLM service fixture."""
    service = GroqSLMService()
    yield service
    await service.close()

@pytest.mark.services
class TestWeatherService:
    """Test weather service functionality."""
    
    @pytest.mark.asyncio
    async def test_weather_forecast(self, weather_service):
        """Test weather forecast retrieval."""
        forecast = await weather_service.get_forecast("London", days=1)
        assert forecast is not None
        assert "temperature" in forecast[0]
        assert "humidity" in forecast[0]

    @pytest.mark.asyncio
    async def test_service_reconnection(self, weather_service):
        """Test service reconnection after failure."""
        # Simulate connection failure
        weather_service.session.close()
        
        # Service should reconnect automatically
        forecast = await weather_service.get_forecast("London", days=1)
        assert forecast is not None

@pytest.mark.services
class TestAstraDBService:
    """Test Astra DB service functionality."""
    
    def test_database_connection(self, astra_service):
        """Test database connection."""
        assert astra_service.session is not None
        assert astra_service.cluster is not None

    def test_data_persistence(self, astra_service):
        """Test data storage and retrieval."""
        test_data = {
            "temperature": 22.5,
            "humidity": 45.0,
            "timestamp": datetime.now()
        }
        
        # Store data
        astra_service.save_temperature_data(test_data)
        
        # Retrieve data
        result = astra_service.get_temperature_data(limit=1)[0]
        assert abs(result.temperature - test_data["temperature"]) < 0.1
        assert abs(result.humidity - test_data["humidity"]) < 0.1

@pytest.mark.services
class TestGroqService:
    """Test Groq SLM service functionality."""
    
    @pytest.mark.asyncio
    async def test_optimization_request(self, groq_service):
        """Test optimization request generation."""
        hvac_data = {
            "temperature": 23.5,
            "humidity": 55.0,
            "power": 1000.0
        }
        
        result = await groq_service.generate_hvac_optimization(
            hvac_data=hvac_data,
            optimization_target="energy_efficiency"
        )
        
        assert "recommendations" in result
        assert "expected_savings" in result
        assert "confidence_score" in result

    @pytest.mark.asyncio
    async def test_response_caching(self, groq_service):
        """Test response caching mechanism."""
        hvac_data = {"temperature": 22.0}
        
        # First request
        result1 = await groq_service.generate_hvac_optimization(
            hvac_data=hvac_data,
            optimization_target="comfort"
        )
        
        # Second request (should be cached)
        result2 = await groq_service.generate_hvac_optimization(
            hvac_data=hvac_data,
            optimization_target="comfort"
        )
        
        assert result1 == result2
