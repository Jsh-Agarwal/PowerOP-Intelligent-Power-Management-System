import aiohttp
import asyncio
import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from functools import wraps
from dotenv import load_dotenv
from cachetools import TTLCache
from ratelimit import limits, sleep_and_retry
from utils.exceptions import WeatherServiceError  # Use absolute import

logger = logging.getLogger('weather_service')

class WeatherRateLimitError(WeatherServiceError):
    """Raised when API rate limit is exceeded."""
    pass

class WeatherAPIError(WeatherServiceError):
    """Raised when API request fails."""
    pass

class WeatherService:
    """Service for interacting with OpenWeatherMap API."""
    
    def __init__(self):
        """Initialize weather service."""
        load_dotenv()
        self.api_key = os.getenv('WEATHER_API_KEY')
        if not self.api_key:
            raise WeatherServiceError("OpenWeatherMap API key not found")
            
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.session = None
        self.cache = {}
        self.cache_ttl = 1800  # 30 minutes

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def get_current_weather(
        self,
        location: str,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """
        Get current weather for a location.
        
        Args:
            location: City name or coordinates
            units: Temperature units (metric/imperial)
            
        Returns:
            Dictionary containing weather data
        """
        try:
            cache_key = f"current_{location}_{units}"
            cached = self._get_from_cache(cache_key)
            if cached:
                return cached

            session = await self.get_session()
            params = {
                "q": location,
                "appid": self.api_key,
                "units": units
            }
            
            async with session.get(
                f"{self.base_url}/weather",
                params=params
            ) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise WeatherServiceError(
                        f"OpenWeatherMap API error: {error_data.get('message', 'Unknown error')}"
                    )
                    
                data = await response.json()
                processed_data = self._process_weather_data(data)
                self._add_to_cache(cache_key, processed_data)
                return processed_data
                
        except Exception as e:
            raise WeatherServiceError(f"Failed to get current weather: {str(e)}")

    async def get_forecast(self, location: str, days: int = 5) -> dict:
        """Get weather forecast for location."""
        try:
            session = await self.get_session()
            cache_key = f"forecast_{location}_{days}"
            
            # Check cache first
            cached = self._get_from_cache(cache_key)
            if cached:
                return cached
                
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric",
                "cnt": days * 8  # API returns data in 3-hour steps
            }
            
            async with session.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=30  # Add timeout
            ) as response:
                if response.status == 404:
                    logger.warning(f"No forecast found for location: {location}")
                    return None
                    
                response.raise_for_status()
                data = await response.json()
                
                if "list" not in data:
                    logger.error(f"Invalid forecast data received: {data}")
                    raise WeatherServiceError("Invalid forecast data format")
                    
                processed = self._process_forecast_data(data)
                self._add_to_cache(cache_key, processed)
                return processed
                
        except asyncio.TimeoutError:
            raise WeatherServiceError("Forecast request timed out")
        except Exception as e:
            logger.error("Forecast request failed", exc_info=True)
            raise WeatherServiceError(f"Forecast failed: {str(e)}")

    def _process_weather_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw weather data into standardized format."""
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "weather_condition": data["weather"][0]["main"],
            "weather_description": data["weather"][0]["description"],
            "timestamp": datetime.fromtimestamp(data["dt"]),
            "location": {
                "name": data["name"],
                "country": data["sys"]["country"],
                "coordinates": {
                    "lat": data["coord"]["lat"],
                    "lon": data["coord"]["lon"]
                }
            }
        }

    def _process_forecast_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw forecast data into standardized format."""
        forecasts = []
        for item in data["list"]:
            forecasts.append({
                "temperature": item["main"]["temp"],
                "humidity": item["main"]["humidity"],
                "pressure": item["main"]["pressure"],
                "wind_speed": item["wind"]["speed"],
                "weather_condition": item["weather"][0]["main"],
                "weather_description": item["weather"][0]["description"],
                "timestamp": datetime.fromtimestamp(item["dt"]),
                "precipitation_probability": item.get("pop", 0) * 100
            })
            
        return {
            "location": {
                "name": data["city"]["name"],
                "country": data["city"]["country"],
                "coordinates": {
                    "lat": data["city"]["coord"]["lat"],
                    "lon": data["city"]["coord"]["lon"]
                }
            },
            "forecasts": forecasts
        }

    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache if not expired."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return data
            del self.cache[key]
        return None

    def _add_to_cache(self, key: str, data: Dict[str, Any]):
        """Add data to cache with current timestamp."""
        self.cache[key] = (data, datetime.now())

    async def test_connection(self) -> bool:
        """Test API connection."""
        try:
            await self.get_current_weather("London")
            return True
        except Exception:
            return False

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def __del__(self):
        """Cleanup on deletion."""
        if self.session and not self.session.closed:
            asyncio.create_task(self.session.close())
