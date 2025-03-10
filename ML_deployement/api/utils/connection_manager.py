import asyncio
from typing import Any, Callable
from functools import wraps
import aiohttp
import backoff
from asyncio import TimeoutError

class ConnectionPool:
    _instances = {}
    _max_connections = 10
    _timeout = 30
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.connections = asyncio.Queue(maxsize=self._max_connections)
        self.active_connections = 0
        
    async def get_connection(self):
        if self.active_connections < self._max_connections:
            self.active_connections += 1
            return await self._create_connection()
        return await self.connections.get()
        
    async def release_connection(self, conn):
        await self.connections.put(conn)
        
    async def _create_connection(self):
        # Implement connection creation based on service
        if self.service_name == "astra":
            from ..services.astra_db_service import AstraDBService
            return await AstraDBService().connect()
        elif self.service_name == "weather":
            from ..services.weather_service import WeatherService
            return await WeatherService().connect()
        # Add other services as needed

def retry_with_backoff(max_tries=3, max_time=30):
    def decorator(func):
        @backoff.on_exception(
            backoff.expo,
            (TimeoutError, ConnectionError),
            max_tries=max_tries,
            max_time=max_time
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                raise e
        return wrapper
    return decorator

class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.state = "closed"
        
    def can_execute(self) -> bool:
        if self.state == "open":
            if (asyncio.get_event_loop().time() - self.last_failure_time) > self.reset_timeout:
                self.state = "half-open"
                return True
            return False
        return True
        
    def record_success(self):
        self.failure_count = 0
        self.state = "closed"
        
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
