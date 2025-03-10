from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict
from ..services.astra_db_service import AstraDBService
from ..services.weather_service import WeatherService
from ..services.models.lstm_model import LSTMModel
from datetime import datetime
from ..utils.error_handlers import handle_api_error
from ..dependencies import get_current_user
from ..models import User
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System"])

@router.get("/api/health")
async def health_check(current_user: User = Depends(get_current_user)):
    """Health check endpoint with authentication."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "user": current_user.username if current_user else None
    }

@router.get("/health")
async def basic_health_check():
    """Basic health check endpoint without authentication."""
    return {"status": "healthy"}

@router.get("/health/authenticated")
async def authenticated_health_check(current_user: User = Depends(get_current_user)):
    """Health check endpoint that requires authentication."""
    return {
        "status": "healthy",
        "authenticated": True,
        "user": current_user
    }
