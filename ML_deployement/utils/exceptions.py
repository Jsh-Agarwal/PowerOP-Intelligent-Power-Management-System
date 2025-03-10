from typing import Dict, Optional, List, Union

class HVACSystemError(Exception):
    """Base exception for HVAC system errors."""
    def __init__(self, message: str = "An error occurred in the HVAC system"):
        self.message = message
        super().__init__(self.message)

class DataProcessingError(HVACSystemError):
    """Raised when data processing operations fail."""
    def __init__(self, message: str = "Data processing error", details: str = None):
        self.details = details
        super_message = f"{message}: {details}" if details else message
        super().__init__(super_message)

class ConfigurationError(HVACSystemError):
    """Raised when configuration validation fails."""
    def __init__(self, message: str = "Configuration error", missing_fields: list = None):
        self.missing_fields = missing_fields
        if missing_fields:
            message = f"{message} - Missing fields: {', '.join(missing_fields)}"
        super().__init__(message)

class ModelError(HVACSystemError):
    """Raised when ML model operations fail."""
    def __init__(self, message: str = "Model error", model_name: str = None):
        self.model_name = model_name
        if model_name:
            message = f"{message} in model {model_name}"
        super().__init__(message)

class ValidationError(HVACSystemError):
    """Raised when data validation fails."""
    def __init__(self, message: str = "Validation error", invalid_fields: list = None):
        self.invalid_fields = invalid_fields
        if invalid_fields:
            message = f"{message} - Invalid fields: {', '.join(invalid_fields)}"
        super().__init__(message)

class SensorError(HVACSystemError):
    """Raised when sensor data is invalid or missing."""
    def __init__(self, sensor_id: str, message: str = "Sensor error"):
        self.sensor_id = sensor_id
        super().__init__(f"{message} for sensor {sensor_id}")

class PerformanceError(HVACSystemError):
    """Raised when system performance is outside expected parameters."""
    def __init__(self, metric: str, value: float, threshold: float):
        self.metric = metric
        self.value = value
        self.threshold = threshold
        message = f"Performance metric {metric} ({value}) exceeded threshold ({threshold})"
        super().__init__(message)

class SystemStateError(HVACSystemError):
    """Raised when system enters an invalid state."""
    def __init__(self, current_state: str, expected_state: str = None):
        self.current_state = current_state
        self.expected_state = expected_state
        message = f"Invalid system state: {current_state}"
        if expected_state:
            message += f", expected: {expected_state}"
        super().__init__(message)

class OptimizationError(HVACSystemError):
    """Raised when optimization operations fail."""
    pass

class ComfortOptimizationError(OptimizationError):
    """Raised when comfort optimization fails."""
    def __init__(self, message: str, metrics: Optional[Dict] = None):
        self.metrics = metrics
        super().__init__(f"Comfort optimization failed: {message}")

class EnergyOptimizationError(OptimizationError):
    """Raised when energy optimization fails."""
    def __init__(self, message: str, metrics: Optional[Dict] = None):
        self.metrics = metrics
        super().__init__(f"Energy optimization failed: {message}")

class PeakLoadError(EnergyOptimizationError):
    """Raised when peak load management fails."""
    def __init__(self, current_load: float, threshold: float):
        self.current_load = current_load
        self.threshold = threshold
        super().__init__(
            f"Peak load ({current_load:.2f} kW) exceeds threshold ({threshold:.2f} kW)"
        )

class RealTimeProcessingError(HVACSystemError):
    """Raised when real-time processing fails."""
    def __init__(self, message: str, data: Optional[Dict] = None):
        self.data = data
        super().__init__(f"Real-time processing failed: {message}")

class WebSocketError(RealTimeProcessingError):
    """Raised when WebSocket operations fail."""
    def __init__(self, client_id: str, message: str):
        self.client_id = client_id
        super().__init__(f"WebSocket error for client {client_id}: {message}")

class WeatherServiceError(HVACSystemError):
    """Raised when weather service operations fail."""
    def __init__(self, message: str, response: Optional[Dict] = None):
        self.response = response
        super().__init__(f"Weather service error: {message}")

class AstraConnectionError(HVACSystemError):
    """Raised when Astra DB connection fails."""
    def __init__(self, message: str):
        super().__init__(f"Astra DB connection error: {message}")

class AstraQueryError(HVACSystemError):
    """Raised when Astra DB query fails."""
    def __init__(self, operation: str, error: Exception):
        self.operation = operation
        self.original_error = error
        super().__init__(f"Astra DB query error in {operation}: {str(error)}")

class AstraDBStateError(HVACSystemError):
    """Raised when Astra DB is in an invalid state."""
    def __init__(self, state: str, message: str):
        self.state = state
        super().__init__(f"Astra DB state error ({state}): {message}")
