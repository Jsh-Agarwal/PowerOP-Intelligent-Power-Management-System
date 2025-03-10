from typing import Dict, Any, Optional
from datetime import datetime
import logging
from .astra_db_service import AstraDBService
from ..utils.error_handlers import APIError

logger = logging.getLogger(__name__)

class SystemController:
    def __init__(self, db_service: AstraDBService):
        self.db = db_service
        
    async def set_temperature(
        self,
        system_id: str,
        temperature: float,
        mode: str = "cool"
    ) -> Dict[str, Any]:
        """Set temperature for a system."""
        try:
            # Validate temperature range
            if temperature < 16 or temperature > 30:
                raise APIError(
                    status_code=400,
                    detail="Temperature must be between 16°C and 30°C",
                    error_code="INVALID_TEMPERATURE"
                )
                
            # Record the command
            command = {
                "system_id": system_id,
                "type": "temperature_control",
                "parameters": {
                    "temperature": temperature,
                    "mode": mode
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in database
            await self.db.store_command(command)
            
            return {
                "status": "success",
                "command": command,
                "message": f"Temperature set to {temperature}°C in {mode} mode"
            }
            
        except Exception as e:
            logger.error(f"Temperature control failed: {str(e)}")
            raise APIError(
                status_code=500,
                detail=f"Failed to set temperature: {str(e)}",
                error_code="CONTROL_ERROR"
            )
            
    async def set_power(
        self,
        system_id: str,
        state: bool
    ) -> Dict[str, Any]:
        """Set power state for a system."""
        try:
            command = {
                "system_id": system_id,
                "type": "power",
                "value": state,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.db.store_command(command)
            
            return {
                "status": "success",
                "command": command,
                "message": f"System {'powered on' if state else 'powered off'}"
            }
            
        except Exception as e:
            raise APIError(
                status_code=500,
                detail=f"Failed to set power state: {str(e)}",
                error_code="CONTROL_ERROR"
            )
            
    async def adjust_temperature(
        self,
        system_id: str,
        step: float
    ) -> Dict[str, Any]:
        """Adjust temperature by step value."""
        try:
            # Get current temperature
            current = await self.db.get_current_temperature(system_id)
            new_temp = round(current + step, 1)
            
            # Set new temperature
            result = await self.set_temperature(
                system_id=system_id,
                temperature=new_temp
            )
            
            return result
            
        except Exception as e:
            raise APIError(
                status_code=500,
                detail=f"Failed to adjust temperature: {str(e)}",
                error_code="CONTROL_ERROR"
            )
