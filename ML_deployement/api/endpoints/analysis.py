from fastapi import APIRouter, Depends, HTTPException, Query, Body
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse
import logging
import json

from ..auth import oauth2_scheme, validate_token
from ..utils.error_handlers import handle_api_error
from ..services.cost_analyzer import CostAnalyzer
from ..services.groq_slm_service import GroqSLMService
from ..utils.json_encoder import DateTimeEncoder
from ..services.models.autoencoder import AutoEncoderModel
from ..schemas import LLMAnalysisRequest, LLMAnalysisResponse, AnomalyDetectionRequest, AnomalyDetectionResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analysis",
    tags=["System Analysis"]
)

@router.get("/temperature/daily/{system_id}")
async def get_daily_temperature(
    system_id: str,
    date: datetime,
    token: str = Depends(oauth2_scheme)
):
    """Get daily temperature analysis."""
    try:
        await validate_token(token)
        return {
            "system_id": system_id,
            "date": date.isoformat(),
            "analysis": {
                "average_temp": 23.5,
                "min_temp": 21.0,
                "max_temp": 25.0,
                "efficiency_score": 0.85
            }
        }
    except Exception as e:
        raise handle_api_error(e, "get_daily_temperature")

@router.get("/cost/{system_id}")
async def get_cost_analysis(
    system_id: str,
    start_time: datetime,
    end_time: datetime,
    token: str = Depends(oauth2_scheme)
):
    """Get cost analysis for a system."""
    try:
        await validate_token(token)
        return {
            "system_id": system_id,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "analysis": {
                "total_cost": 125.50,
                "energy_usage": 500.0,
                "savings_potential": 15.0
            }
        }
    except Exception as e:
        raise handle_api_error(e, "get_cost_analysis")

@router.post("/optimize/llm/{system_id}", response_model=LLMAnalysisResponse)
async def analyze_with_llm(
    system_id: str,
    request: LLMAnalysisRequest,
    token: str = Depends(oauth2_scheme)
):
    """
    Analyze system using Groq LLM.
    """
    try:
        await validate_token(token)
        groq = GroqSLMService()
        await groq.connect()
        
        result = await groq.analyze(
            system_id=system_id,
            query=request.query,
            context=request.context.model_dump()
        )
        
        return {
            "status": "success",
            "data": result
        }
            
    except Exception as e:
        raise handle_api_error(e, "analyze_with_llm")

@router.post("/anomaly/detect/{system_id}", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    system_id: str,
    request: AnomalyDetectionRequest,
    token: str = Depends(oauth2_scheme)
):
    """Detect anomalies in system data."""
    try:
        await validate_token(token)
        
        # Process anomaly detection
        anomalies = [
            {
                "timestamp": point["timestamp"],
                "metric": "temperature",
                "value": point["temperature"],
                "confidence": 0.95,
                "type": "outlier"
            }
            for point in request.data
            if abs(point["temperature"] - 23.5) > 5.0  # Simple threshold check
        ]
        
        return {
            "system_id": system_id,
            "anomalies": anomalies,
            "summary": {
                "total_points": len(request.data),
                "anomalies_found": len(anomalies),
                "confidence_threshold": request.threshold
            }
        }
    except Exception as e:
        raise handle_api_error(e, "detect_anomalies")