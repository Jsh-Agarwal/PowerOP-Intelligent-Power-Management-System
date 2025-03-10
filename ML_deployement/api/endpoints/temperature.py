from fastapi import APIRouter, Depends, HTTPException, Security, Body, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from fastapi.responses import JSONResponse

from ..schemas import TemperaturePredictionRequest, TemperaturePredictionResponse, BatchPredictionRequest
from ..services.models.lstm_model import LSTMModel
from ..services.astra_db_service import AstraDBService
from ..utils.exceptions import ModelError, ValidationError
from ..auth import validate_token, oauth2_scheme
from ..services.groq_slm_service import GroqSLMService
from models.model_manager import ModelManager  # Fix import path
from ..utils.error_handlers import handle_api_error

router = APIRouter(
    prefix="/api/temperature",  # Keep this prefix
    tags=["Temperature Management"]
)

# Initialize logger
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=TemperaturePredictionResponse)
async def predict_temperature(request: TemperaturePredictionRequest, token: str = Depends(oauth2_scheme)):
    """Generate temperature predictions."""
    try:
        await validate_token(token)
        
        # Process prediction
        model = LSTMModel()
        predictions = await model.predict(
            features=request.features
        )
        
        # Generate timestamps
        now = datetime.utcnow()
        timestamps = [now + timedelta(hours=i) for i in range(len(predictions["predictions"]))]
            
        return TemperaturePredictionResponse(
            predictions=predictions["predictions"],
            timestamps=timestamps,
            confidence=predictions.get("confidence")
        )
    except Exception as e:
        logger.error(f"Temperature prediction failed: {str(e)}")
        raise handle_api_error(e, "predict_temperature")

@router.post("/train")
async def train_temperature_model(
    request: Dict[str, Any] = Body(...),
    token: str = Depends(oauth2_scheme)
):
    """Train temperature prediction model."""
    try:
        await validate_token(token)
        
        if not request.get("features"):
            request["features"] = {
                "temperature": 23.5,
                "humidity": 50.0,
                "time_of_day": 8.0
            }
            
        model_manager = ModelManager()
        result = await model_manager.train_model(request)
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "model_info": result
        }
    except Exception as e:
        raise handle_api_error(e, "train_temperature_model")

@router.get("/history")
async def get_temperature_history(
    device_id: str = Query(..., description="Device identifier"),
    zone_id: str = Query(..., description="Zone identifier"),
    start_time: Optional[datetime] = Query(None, description="Start timestamp"),
    end_time: Optional[datetime] = Query(None, description="End timestamp"),
    token: str = Depends(oauth2_scheme)
):
    """Get historical temperature data."""
    try:
        await validate_token(token)
        
        db = AstraDBService()
        data = await db.get_temperature_data(
            device_id=device_id,
            zone_id=zone_id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Convert data to a format that can be JSON serialized
        history = [
            {
                "temperature": reading["temperature"],
                "humidity": reading["humidity"],
                "timestamp": reading["timestamp"].isoformat(),
                "device_id": device_id,
                "zone_id": zone_id
            }
            for reading in data
        ]
        
        return {
            "history": history,
            "device_id": device_id,
            "zone_id": zone_id,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "count": len(history)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to fetch history", "message": str(e)}
        )

@router.get("/current")
async def get_current_temperature(
    device_id: str = Query(..., description="Device identifier"),
    zone_id: str = Query(..., description="Zone identifier"),
    token: str = Depends(oauth2_scheme)
):
    """Get real-time temperature data."""
    try:
        await validate_token(token)
        db = AstraDBService()
        
        try:
            data = await db.get_temperature_data(
                device_id=device_id,
                zone_id=zone_id,
                limit=1
            )
            
            if not data:
                return JSONResponse(
                    status_code=404,
                    content={"detail": "No current data available"}
                )
            
            reading = data[0]
            return {
                "temperature": reading["temperature"],
                "humidity": reading["humidity"],
                "timestamp": reading["timestamp"].isoformat(),
                "device_id": device_id,
                "zone_id": zone_id
            }
            
        finally:
            await db.close()
            
    except Exception as e:
        raise handle_api_error(e, "get_current_temperature")

@router.post("/batch")
async def batch_predict_temperature(request: BatchPredictionRequest, token: str = Depends(oauth2_scheme)):
    """Process batch of temperature prediction requests."""
    model = None
    try:
        payload = await validate_token(token)
        
        if not request.predictions:
            raise HTTPException(
                status_code=422,
                detail="Batch requests cannot be empty"
            )
            
        # Process batch predictions
        results = []
        errors = []
        model = LSTMModel()
        
        for idx, pred_request in enumerate(request.predictions):
            try:
                result = await model.predict(
                    features=pred_request.features,
                    device_id=pred_request.device_id,
                    zone_id=pred_request.zone_id
                )
                results.append({
                    "id": idx,
                    "predictions": result["predictions"],
                    "device_id": pred_request.device_id,
                    "zone_id": pred_request.zone_id
                })
            except Exception as e:
                errors.append({
                    "id": idx,
                    "error": str(e),
                    "device_id": pred_request.device_id,
                    "zone_id": pred_request.zone_id
                })
                
        return {
            "status": "success",
            "results": results,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        raise handle_api_error(e, "batch_predict_temperature")
    finally:
        if model:
            await model.close()
