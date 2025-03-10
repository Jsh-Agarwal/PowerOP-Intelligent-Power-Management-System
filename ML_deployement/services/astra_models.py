from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from uuid import UUID, uuid4

@dataclass
class TemperatureData:
    """Temperature readings from HVAC system sensors."""
    timestamp: datetime
    device_id: str
    zone_id: str
    temperature: float
    humidity: Optional[float] = None
    id: UUID = field(default_factory=uuid4)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "zone_id": self.zone_id,
            "temperature": self.temperature,
            "humidity": self.humidity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemperatureData':
        """Create instance from dictionary."""
        return cls(
            id=data.get("id", uuid4()),
            timestamp=data["timestamp"],
            device_id=data["device_id"],
            zone_id=data["zone_id"],
            temperature=data["temperature"],
            humidity=data.get("humidity")
        )

@dataclass
class SystemStatus:
    """HVAC system operational status."""
    timestamp: datetime
    system_id: str
    status: str  # "running", "idle", "error", etc.
    mode: str  # "heating", "cooling", "fan", etc.
    active_power: float
    energy_consumption: float
    pressure_high: float
    pressure_low: float
    error_code: Optional[str] = None
    id: UUID = field(default_factory=uuid4)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "system_id": self.system_id,
            "status": self.status,
            "mode": self.mode,
            "active_power": self.active_power,
            "energy_consumption": self.energy_consumption,
            "pressure_high": self.pressure_high,
            "pressure_low": self.pressure_low,
            "error_code": self.error_code
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemStatus':
        """Create instance from dictionary."""
        return cls(
            id=data.get("id", uuid4()),
            timestamp=data["timestamp"],
            system_id=data["system_id"],
            status=data["status"],
            mode=data["mode"],
            active_power=data["active_power"],
            energy_consumption=data["energy_consumption"],
            pressure_high=data["pressure_high"],
            pressure_low=data["pressure_low"],
            error_code=data.get("error_code")
        )

@dataclass
class UserPreference:
    """User preference settings for HVAC system."""
    user_id: str
    zone_id: str
    preferred_temperature: float
    preferred_humidity: Optional[float] = None
    schedule: Dict[str, Any] = field(default_factory=dict)
    mode_preference: str = "auto"  # "auto", "heat", "cool", "fan"
    energy_saving: bool = False
    priority: int = 1  # 1-high, 5-low
    last_updated: datetime = field(default_factory=datetime.now)
    id: UUID = field(default_factory=uuid4)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "zone_id": self.zone_id,
            "preferred_temperature": self.preferred_temperature,
            "preferred_humidity": self.preferred_humidity,
            "schedule": self.schedule,
            "mode_preference": self.mode_preference,
            "energy_saving": self.energy_saving,
            "priority": self.priority,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreference':
        """Create instance from dictionary."""
        return cls(
            id=data.get("id", uuid4()),
            user_id=data["user_id"],
            zone_id=data["zone_id"],
            preferred_temperature=data["preferred_temperature"],
            preferred_humidity=data.get("preferred_humidity"),
            schedule=data.get("schedule", {}),
            mode_preference=data.get("mode_preference", "auto"),
            energy_saving=data.get("energy_saving", False),
            priority=data.get("priority", 1),
            last_updated=data.get("last_updated", datetime.now())
        )

@dataclass
class OptimizationResult:
    """Results from optimization algorithms."""
    timestamp: datetime
    system_id: str
    optimization_type: str  # "energy", "comfort", "cost"
    recommendations: List[Dict[str, Any]]
    expected_savings: float
    confidence_score: float
    applied: bool = False
    result_id: UUID = field(default_factory=uuid4)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return {
            "result_id": self.result_id,
            "timestamp": self.timestamp,
            "system_id": self.system_id,
            "optimization_type": self.optimization_type,
            "recommendations": self.recommendations,
            "expected_savings": self.expected_savings,
            "confidence_score": self.confidence_score,
            "applied": self.applied
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationResult':
        """Create instance from dictionary."""
        return cls(
            result_id=data.get("result_id", uuid4()),
            timestamp=data["timestamp"],
            system_id=data["system_id"],
            optimization_type=data["optimization_type"],
            recommendations=data["recommendations"],
            expected_savings=data["expected_savings"],
            confidence_score=data["confidence_score"],
            applied=data.get("applied", False)
        )

@dataclass
class AnomalyEvent:
    """Anomalies detected by the autoencoder."""
    timestamp: datetime
    system_id: str
    metric: str
    value: float
    threshold: float
    severity: str  # "critical", "high", "medium", "low"
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    event_id: UUID = field(default_factory=uuid4)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "system_id": self.system_id,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "severity": self.severity,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnomalyEvent':
        """Create instance from dictionary."""
        return cls(
            event_id=data.get("event_id", uuid4()),
            timestamp=data["timestamp"],
            system_id=data["system_id"],
            metric=data["metric"],
            value=data["value"],
            threshold=data["threshold"],
            severity=data["severity"],
            resolved=data.get("resolved", False),
            resolution_time=data.get("resolution_time")
        )
