from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from ..auth import oauth2_scheme, validate_token
from ..utils.error_handlers import handle_api_error
from ..services.system_controller import SystemController
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/control",
    tags=["System Control"]
)

class TemperatureControl(BaseModel):
    system_id: str
    temperature: float
    mode: Optional[str] = "cool"  # cool, heat, auto

class PowerControl(BaseModel):
    system_id: str
    state: bool  # True for on, False for off

class ControlRequest(BaseModel):
    system_id: str
    temperature: Optional[float] = None
    mode: Optional[str] = "cool"
    state: Optional[bool] = None

@router.post("/temperature")
async def set_temperature(
    request: ControlRequest,
    token: str = Depends(oauth2_scheme)
):
    """Set system temperature."""
    try:
        await validate_token(token)
        
        if not request.temperature:
            raise HTTPException(status_code=422, detail="Temperature is required")
            
        # Implement temperature control logic
        return {
            "status": "success",
            "system_id": request.system_id,
            "temperature": request.temperature,
            "mode": request.mode
        }
    except Exception as e:
        raise handle_api_error(e, "set_temperature")

@router.post("/power")
async def set_power_state(
    request: ControlRequest,
    token: str = Depends(oauth2_scheme)
):
    """Set system power state."""
    try:
        await validate_token(token)
        
        if request.state is None:
            raise HTTPException(status_code=422, detail="Power state is required")
            
        return {
            "status": "success",
            "system_id": request.system_id,
            "power_state": request.state
        }
    except Exception as e:
        raise handle_api_error(e, "set_power_state")

@router.post("/temperature/increment/{system_id}")
async def increment_temperature(
    system_id: str,
    token: str = Depends(oauth2_scheme)
):
    """Increment system temperature."""
    try:
        await validate_token(token)
        # Implement temperature increment logic
        return {
            "status": "success",
            "system_id": system_id,
            "action": "increment"
        }
    except Exception as e:
        raise handle_api_error(e, "increment_temperature")

@router.post("/temperature/decrement/{system_id}")
async def decrement_temperature(
    system_id: str,
    token: str = Depends(oauth2_scheme)
):
    """Decrement system temperature."""
    try:
        await validate_token(token)
        # Implement temperature decrement logic
        return {
            "status": "success",
            "system_id": system_id,
            "action": "decrement"
        }
    except Exception as e:
        raise handle_api_error(e, "decrement_temperature")
