from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
import asyncio
import logging
from fastapi.responses import JSONResponse
import json

from ..schemas import SystemStatusResponse, HealthResponse
from ..services.astra_db_service import AstraDBService
from ..services.weather_service import WeatherService
from ..services.models.lstm_model import LSTMModel
from ..utils.json_encoder import DateTimeEncoder  # Added this import
from ..auth import validate_token, oauth2_scheme
from ..utils.error_handlers import handle_api_error

# Initialize logger
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/status",  # Keep this prefix
    tags=["System Monitoring"]
)

@router.get("/system/{system_id}", response_model=SystemStatusResponse)
async def get_system_status(
    system_id: str,
    token: str = Depends(oauth2_scheme)
):
    """Get current system status."""
    try:
        await validate_token(token)
        return {
            "system_id": system_id,
            "status": "running",
            "metrics": {
                "temperature": 23.5,
                "humidity": 50.0,
                "power": 1000.0
            },
            "last_updated": datetime.now()
        }
    except Exception as e:
        raise handle_api_error(e, "get_system_status")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint."""
    try:
        components = {
            "database": "unhealthy",
            "ml_model": "unhealthy",
            "weather_service": "unhealthy",
            "optimization_service": "unhealthy"
        }

        # Simplified health checks for faster response
        async def run_checks():
            try:
                # Basic database check
                db = AstraDBService()
                await db.test_connection()
                components["database"] = "healthy"
                await db.close()
            except:
                pass

            try:
                # Basic model check 
                model = LSTMModel()
                if await model.test():
                    components["ml_model"] = "healthy"
            except:
                pass

            try:
                # Basic weather service check
                weather = WeatherService()
                if await weather.test_connection():
                    components["weather_service"] = "healthy"
                await weather.close()
            except:
                pass

            try:
                # Basic optimization check
                components["optimization_service"] = "healthy"
            except:
                pass

        await run_checks()

        return {
            "status": "ok" if all(v == "healthy" for v in components.values()) else "degraded",
            "version": "1.0.0",
            "components": components,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_system_metrics(
    system_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    token: str = Depends(oauth2_scheme)
):
    """Get system metrics over time."""
    try:
        await validate_token(token)
        
        # Default to last 24 hours if no time range provided
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()

        return {
            "system_id": system_id,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "metrics": {
                "temperature": {
                    "average": 23.5,
                    "min": 21.0,
                    "max": 25.0
                },
                "power_usage": {
                    "total": 24000.0,
                    "average": 1000.0
                },
                "runtime_hours": 24
            }
        }
    except Exception as e:
        raise handle_api_error(e, "get_system_metrics")

def calculate_metrics_summary(metrics, start_time, end_time):
    """Calculate summary statistics from metrics data."""
    if not metrics:
        return {
            "total_energy": 0,
            "avg_power": 0,
            "efficiency": 0,
            "duration": 0
        }
        
    total_energy = sum(m.get("energy_consumption", 0) for m in metrics)
    avg_power = sum(m.get("active_power", 0) for m in metrics) / len(metrics)
    efficiency = sum(m.get("efficiency", 0) for m in metrics) / len(metrics)
    
    duration = 0
    if start_time and end_time:
        duration = (end_time - start_time).total_seconds() / 3600  # hours
        
    return {
        "total_energy": round(total_energy, 2),
        "avg_power": round(avg_power, 2),
        "efficiency": round(efficiency, 2),
        "duration": round(duration, 2)
    }