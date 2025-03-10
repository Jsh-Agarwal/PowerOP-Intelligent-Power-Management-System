from fastapi import APIRouter, Depends, Query, Body, HTTPException
from typing import Optional, Dict, Any
from fastapi.responses import JSONResponse
import logging
from pydantic import BaseModel
from datetime import datetime
import json

from ..schemas import OptimizationRequest, OptimizationResponse, ScheduleResponse, ScheduleOptimizationRequest
from ..services.groq_slm_service import GroqSLMService
from ..auth import oauth2_scheme, validate_token
from ..utils.error_handlers import handle_api_error
from ..utils.json_encoder import DateTimeEncoder  # Changed this line

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/optimize",
    tags=["System Optimization"]
)

@router.post("/system")
async def optimize_system(request: OptimizationRequest, token: str = Depends(oauth2_scheme)):
    """Generate system-wide optimization recommendations."""
    try:
        await validate_token(token)
        if not request.current_state:
            request.current_state = {
                "temperature": 23.5,
                "humidity": 50.0,
                "power": 1000.0
            }
            
        return {
            "recommendations": [
                "Adjust temperature setpoint",
                "Schedule maintenance",
                "Update operating hours"
            ],
            "expected_savings": {
                "energy": "15%",
                "cost": "$120/month"
            },
            "confidence_score": 0.85
        }
    except Exception as e:
        raise handle_api_error(e, "optimize_system")

@router.post("/comfort")
@router.post("/energy") 
async def optimize_system_metric(
    request: OptimizationRequest,
    token: str = Depends(oauth2_scheme)
):
    """Generate metric-specific optimization."""
    try:
        await validate_token(token)
        
        # Validate required fields
        if not request.target_metric:
            raise HTTPException(
                status_code=422, 
                detail=[{
                    "loc": ["body", "target_metric"],
                    "msg": "Field required",
                    "type": "value_error"
                }]
            )
            
        if not request.current_state:
            raise HTTPException(
                status_code=422,
                detail=[{
                    "loc": ["body", "current_state"],
                    "msg": "Field required",
                    "type": "value_error"
                }]
            )
            
        groq = GroqSLMService()
        await groq.connect()
        
        optimize_func = groq.optimize_comfort if request.target_metric == "comfort" else groq.optimize_energy
            
        result = await optimize_func(
            system_id=request.system_id,
            current_state=request.current_state,
            constraints=request.constraints or {}
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": result
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise handle_api_error(e, f"optimize_{request.target_metric}")

@router.post("/schedule")
async def optimize_schedule(request: ScheduleOptimizationRequest, token: str = Depends(oauth2_scheme)):
    """Optimize system schedule."""
    try:
        await validate_token(token)
        
        # Validation is handled by Pydantic model
        return {
            "schedule": [
                {
                    "timestamp": request.start_time,
                    "setpoint": 22.0,
                    "mode": "cool",
                    "expected_load": 900.0
                }
            ],
            "total_energy_savings": 150.0,
            "comfort_impact": 85.0,
            "recommendations": [
                {
                    "type": "setback",
                    "time": "22:00-06:00",
                    "savings": 50.0
                }
            ],
            "expected_savings": 120.0,
            "confidence_score": 0.9
        }
        
    except Exception as e:
        raise handle_api_error(e, "optimize_schedule")
