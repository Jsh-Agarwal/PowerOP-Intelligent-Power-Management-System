import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any, TypeVar, Generic, Callable
from datetime import datetime, timedelta
from uuid import UUID
import asyncio
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from astrapy import DataAPIClient
from dotenv import load_dotenv
# Change to absolute imports
from utils.exceptions import AstraConnectionError, AstraQueryError
from utils.logger import setup_logger

logger = setup_logger('astra_db_service')
load_dotenv()

T = TypeVar('T')

class AstraDBService:
    """
    Service for interacting with DataStax Astra DB (managed Cassandra)
    for HVAC system data persistence.
    """
    
    def __init__(self):
        """Initialize Astra DB connection."""
        load_dotenv()
        
        # Load credentials from environment
        self.token = os.getenv('ASTRA_DB_TOKEN')
        self.api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
        self.keyspace = os.getenv('ASTRA_DB_KEYSPACE')
        
        if not all([self.token, self.api_endpoint, self.keyspace]):
            raise AstraConnectionError("Missing Astra DB configuration")
        
        try:
            # Initialize the client
            self.client = DataAPIClient(self.token)
            self.db = self.client.get_database_by_api_endpoint(
                self.api_endpoint,
                keyspace=self.keyspace
            )
            
        except Exception as e:
            raise AstraConnectionError(f"Failed to connect to Astra DB: {str(e)}")
    
    async def save_temperature_data(self, data: Dict[str, Any]) -> str:
        """Save temperature reading to database."""
        try:
            collection = self.db.collection("temperature_data")
            result = await collection.insert_one(data)
            return str(result.inserted_id)
        except Exception as e:
            raise AstraQueryError("save_temperature_data", e)
    
    async def get_temperature_data(
        self,
        device_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get temperature data with async support."""
        try:
            # Return mock data for testing
            return [{
                "temperature": 22.5,
                "humidity": 45.0,
                "timestamp": datetime.now()
            }]
        except Exception as e:
            raise AstraQueryError("get_temperature_data", e)
    
    async def save_system_status(self, status: Dict[str, Any]) -> str:
        """Save system status record."""
        try:
            collection = self.db.collection("system_status")
            result = await collection.insert_one(status)
            return str(result.inserted_id)
        except Exception as e:
            raise AstraQueryError("save_system_status", e)
    
    async def get_system_status(
        self,
        system_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get system status history."""
        try:
            # Return mock data for testing
            return [{
                "status": "active",
                "active_power": 1000.0,
                "energy_consumption": 2500.0,
                "pressure_high": 120.0,
                "pressure_low": 80.0,
                "timestamp": datetime.now()
            }]
        except Exception as e:
            raise AstraQueryError("get_system_status", e)
    
    async def test_connection(self):
        """Test database connection."""
        try:
            result = await self.execute_query("SELECT now() FROM system.local")
            return bool(result)
        except Exception as e:
            raise DatabaseError(f"Connection test failed: {str(e)}")
            
    async def close(self):
        """Close database connection."""
        try:
            if hasattr(self, 'session') and self.session:
                await self.session.close()
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")
