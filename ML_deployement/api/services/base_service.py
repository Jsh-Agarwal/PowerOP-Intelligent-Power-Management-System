from abc import ABC, abstractmethod
import logging
from datetime import datetime

class BaseService(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    async def connect(self):
        """Establish connection to service."""
        pass
        
    @abstractmethod
    async def close(self):
        """Close service connection."""
        pass
        
    @abstractmethod
    async def test_connection(self):
        """Test if service is available.""" 
        pass

    async def health_check(self) -> dict:
        """Check if service is healthy and return status details."""
        try:
            is_connected = await self.test_connection()
            return {
                "status": "healthy" if is_connected else "degraded",
                "last_checked": datetime.utcnow().isoformat(),
                "details": {
                    "connected": is_connected,
                    "service_name": self.__class__.__name__
                }
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": datetime.utcnow().isoformat(),
                "details": {
                    "connected": False,
                    "service_name": self.__class__.__name__
                }
            }
