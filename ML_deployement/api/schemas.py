from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from uuid import UUID

class FeatureDict(BaseModel):
    temperature: float
    humidity: float
    time_of_day: float

class TemperaturePredictionRequest(BaseModel):
    device_id: str = Field(..., description="Device identifier")
    zone_id: str = Field(..., description="Zone identifier")
    features: Dict[str, float] = Field(..., description="Feature values")
    current_temp: float = Field(..., description="Current temperature")
    target_temp: float = Field(..., description="Target temperature")
    external_temp: Optional[float] = None
    system_id: Optional[str] = None

    @field_validator('features')
    def validate_features(cls, v):
        required = {'temperature', 'humidity', 'time_of_day'}
        missing = required - v.keys()
        if missing:
            raise ValueError(f'Missing required features: {missing}')
        return v

class TemperaturePredictionResponse(BaseModel):
    predictions: List[float]
    timestamps: List[datetime]
    confidence: Optional[float] = None

class BatchPredictionItem(BaseModel):
    device_id: str
    zone_id: str
    features: Dict[str, float]
    timestamps: Optional[List[datetime]] = None

    @field_validator('features')
    def validate_features(cls, v):
        required = {'temperature', 'humidity', 'time_of_day'}
        missing = required - v.keys()
        if missing:
            raise ValueError(f'Missing required features: {missing}')
        return v

class BatchPredictionRequest(BaseModel):
    predictions: List[TemperaturePredictionRequest] = Field(..., description="List of prediction requests")

    @field_validator('predictions')
    def validate_batch(cls, v):
        if not v:
            raise ValueError("Batch must contain at least one prediction request")
        if len(v) > 100:  # Add reasonable limit
            raise ValueError("Batch size exceeds maximum limit of 100")
        return v

class OptimizationRequest(BaseModel):
    system_id: str = Field(..., description="System identifier")
    target_metric: str = Field(..., description="Optimization target metric", 
                             pattern="^(energy|comfort|cost|efficiency)$")
    current_state: Dict[str, Any] = Field(..., description="Current system state")
    constraints: Optional[Dict[str, Any]] = None

    @field_validator('current_state')
    def validate_current_state(cls, v):
        required = {'temperature', 'humidity', 'power'}
        if not isinstance(v, dict):
            raise ValueError('current_state must be a dictionary')
        missing = required - set(v.keys())
        if missing:
            raise ValueError(f'Missing required state parameters: {missing}')
        return v

    @field_validator('target_metric')
    def validate_target_metric(cls, v):
        allowed = {'energy', 'comfort', 'cost', 'efficiency'}
        if v not in allowed:
            raise ValueError(f'target_metric must be one of {allowed}')
        return v

class ScheduleRequest(BaseModel):
    system_id: str
    target_metric: str
    current_state: Dict[str, float]
    start_time: datetime
    end_time: datetime
    interval: int = 60
    constraints: Optional[Dict[str, Any]] = None

class ScheduleOptimizationRequest(BaseModel):
    system_id: str = Field(..., description="System identifier")
    target_metric: str = Field(..., description="Target metric for optimization")
    current_state: Dict[str, float] = Field(..., description="Current system state")
    start_time: datetime = Field(..., description="Schedule start time")
    end_time: datetime = Field(..., description="Schedule end time")
    interval: int = Field(
        default=60,
        ge=15,
        le=1440,
        description="Schedule interval in minutes"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        default={},
        description="Optimization constraints"
    )

    @field_validator('target_metric')
    def validate_target_metric(cls, v):
        allowed = {'energy', 'comfort', 'cost', 'efficiency'}
        if v not in allowed:
            raise ValueError(f'target_metric must be one of {allowed}')
        return v

    @field_validator('current_state')
    def validate_state(cls, v, info):
        if not isinstance(v, dict):
            raise ValueError('current_state must be a dictionary')
            
        required = {'temperature', 'humidity', 'power'}
        missing = required - v.keys()
        if missing:
            raise ValueError(f'Missing required state parameters: {missing}')
        return v

    @field_validator('end_time')
    def validate_time_range(cls, v, info):
        data = info.data
        if 'start_time' in data and v <= data['start_time']:
            raise ValueError('end_time must be after start_time')
        return v

class OptimizationResponse(BaseModel):
    recommendations: List[str] = Field(..., description="Optimization recommendations")
    expected_savings: Dict[str, str] = Field(..., description="Expected savings")
    confidence_score: float = Field(..., description="Confidence score")

class ScheduleEntry(BaseModel):
    timestamp: datetime
    setpoint: float
    mode: str
    expected_load: float

class ScheduleResponse(BaseModel):
    schedule: List[ScheduleEntry] = Field(..., description="Optimized schedule entries")
    total_energy_savings: float = Field(..., description="Total expected energy savings")
    comfort_impact: float = Field(
        ..., 
        description="Expected impact on comfort (0-100)", 
        ge=0, 
        le=100
    )
    recommendations: List[Dict[str, Union[str, float]]] = Field(..., description="Schedule recommendations")
    expected_savings: float = Field(..., description="Expected savings")
    confidence_score: float = Field(..., description="Confidence score")

class SystemStatusResponse(BaseModel):
    system_id: str = Field(..., description="System ID")
    status: str = Field(..., description="Current status")
    metrics: Dict[str, float] = Field(..., description="Current metrics")
    last_updated: datetime = Field(..., description="Last update timestamp")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component statuses")
    timestamp: str = Field(..., description="Timestamp")

class AnomalyDetectionRequest(BaseModel):
    system_id: str
    data: List[Dict[str, Any]] = Field(..., description="Time series data points")
    threshold: float = Field(
        default=0.95,
        gt=0,
        le=1,
        description="Anomaly detection threshold"
    )

    model_config = {
        'json_schema_extra': {
            "example": {
                "system_id": "test_system",
                "data": [{
                    "timestamp": "2025-03-01T00:00:00Z",
                    "temperature": 23.5,
                    "humidity": 50.0,
                    "power": 1000.0,
                    "pressure": 101.3
                }],
                "threshold": 0.95
            }
        }
    }

class AnomalyDetectionResponse(BaseModel):
    system_id: str
    anomalies: List[Dict[str, Any]]
    summary: Dict[str, Any]

class DailyAnalysisRequest(BaseModel):
    system_id: str
    date: Optional[datetime] = None

class CostAnalysisRequest(BaseModel):
    system_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class LLMAnalysisContext(BaseModel):
    """Context data for LLM analysis."""
    temperature: Optional[float] = Field(None, description="Current temperature")
    power: Optional[float] = Field(None, description="Current power consumption")
    runtime_hours: Optional[int] = Field(None, description="Runtime hours")
    
    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'temperature': 23.5,
                    'power': 1000.0,
                    'runtime_hours': 24
                }
            ]
        }
    }

class LLMAnalysisRequest(BaseModel):
    """Request model for LLM analysis."""
    query: str = Field(..., description="Analysis query")
    context: LLMAnalysisContext

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'query': "Analyze system efficiency",
                    'context': {
                        'temperature': 23.5,
                        'power': 1000.0,
                        'runtime_hours': 24
                    }
                }
            ]
        }
    }

class LLMAnalysisResponse(BaseModel):
    """Response model for LLM analysis."""
    status: str = Field(..., description="Analysis status")
    data: Dict[str, Any] = Field(..., description="Analysis results")

class ControlRequest(BaseModel):
    system_id: str = Field(..., description="System identifier")
    temperature: Optional[float] = Field(None, description="Temperature setpoint")
    mode: Optional[str] = Field(None, description="Operation mode")
    state: Optional[bool] = Field(None, description="Power state")
