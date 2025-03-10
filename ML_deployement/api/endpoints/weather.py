from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from ..utils.json_encoder import DateTimeEncoder
import json

from ..services.weather_service import WeatherService
from ..auth import oauth2_scheme, validate_token
from ..utils.error_handlers import handle_api_error, APIError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/weather",
    tags=["Weather"]
)

@router.get("/current")
async def get_current_weather(
    location: str = Query(..., description="Location to get weather for"),
    token: str = Depends(oauth2_scheme)
):
    """Get current weather data for a location."""
    try:
        await validate_token(token)
        weather_service = WeatherService()
        data = await weather_service.get_current_weather(location)
        return {
            "location": location,
            "current_conditions": data
        }
    except Exception as e:
        raise handle_api_error(e, "get_current_weather")

@router.get("/forecast")
async def get_weather_forecast(
    location: str = Query(..., description="Location to get forecast for"),
    days: Optional[int] = Query(5, description="Number of days to forecast"),
    token: str = Depends(oauth2_scheme)
):
    """Get weather forecast for a location."""
    try:
        await validate_token(token)
        weather_service = WeatherService()
        forecast = await weather_service.get_forecast(location, days)
        return {
            "location": location,
            "forecast_days": days,
            "forecast": forecast
        }
    except Exception as e:
        raise handle_api_error(e, "get_weather_forecast")
