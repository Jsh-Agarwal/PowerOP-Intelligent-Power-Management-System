from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List, Set, Optional, Any
import asyncio
import json
import logging
from datetime import datetime
import numpy as np
from dataclasses import dataclass

from ..services.astra_db_service import AstraDBService
from ..models.autoencoder import Autoencoder
from ..utils.exceptions import RealTimeProcessingError
from ..utils.logger import setup_logger

logger = setup_logger('real_time_processor')

@dataclass
class AlertConfig:
    """Alert configuration settings."""
    threshold_multiplier: float = 1.5
    min_consecutive_anomalies: int = 3
    cooldown_period: int = 300  # seconds
    alert_levels: Dict[str, float] = None

    def __post_init__(self):
        if self.alert_levels is None:
            self.alert_levels = {
                "warning": 1.5,
                "critical": 2.0,
                "emergency": 3.0
            }

class ConnectionManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.client_info: Dict[WebSocket, Dict[str, Any]] = {}
        
    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        system_id: str
    ):
        """Connect new client."""
        await websocket.accept()
        if system_id not in self.active_connections:
            self.active_connections[system_id] = set()
        self.active_connections[system_id].add(websocket)
        self.client_info[websocket] = {
            "client_id": client_id,
            "system_id": system_id,
            "connected_at": datetime.now()
        }
        
    def disconnect(self, websocket: WebSocket):
        """Disconnect client."""
        system_id = self.client_info[websocket]["system_id"]
        self.active_connections[system_id].remove(websocket)
        if not self.active_connections[system_id]:
            del self.active_connections[system_id]
        del self.client_info[websocket]
        
    async def broadcast_to_system(
        self,
        system_id: str,
        message: Dict[str, Any]
    ):
        """Broadcast message to all clients of a system."""
        if system_id not in self.active_connections:
            return
            
        dead_connections = set()
        for connection in self.active_connections[system_id]:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.add(connection)
                
        # Clean up dead connections
        for dead in dead_connections:
            self.disconnect(dead)

class RealTimeProcessor:
    """Real-time data processing and monitoring system."""
    
    def __init__(
        self,
        db_service: Optional[AstraDBService] = None,
        autoencoder: Optional[Autoencoder] = None,
        alert_config: Optional[AlertConfig] = None
    ):
        """Initialize real-time processor."""
        self.db = db_service or AstraDBService()
        self.autoencoder = autoencoder
        self.alert_config = alert_config or AlertConfig()
        self.connection_manager = ConnectionManager()
        
        # Processing state
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.anomaly_counters: Dict[str, int] = {}
        self.last_alert_time: Dict[str, datetime] = {}
        self.system_states: Dict[str, Dict[str, Any]] = {}
        
        # Initialize autoencoder if not provided
        if not self.autoencoder:
            self._initialize_autoencoder()

    def _initialize_autoencoder(self):
        """Initialize autoencoder model."""
        try:
            self.autoencoder = Autoencoder(input_dim=10)  # Adjust input_dim as needed
            self.autoencoder.load_model(
                model_path="models/autoencoder.h5",
                scaler_path="models/scaler.joblib",
                threshold_path="models/threshold.npy"
            )
        except Exception as e:
            raise RealTimeProcessingError(f"Failed to initialize autoencoder: {str(e)}")

    async def process_sensor_data(
        self,
        system_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process incoming sensor data."""
        try:
            # Validate and preprocess data
            processed_data = self._preprocess_sensor_data(data)
            
            # Check for anomalies
            anomalies = await self._detect_anomalies(
                system_id,
                processed_data
            )
            
            # Update system state
            self.system_states[system_id] = {
                "current_data": processed_data,
                "anomalies": anomalies,
                "last_update": datetime.now()
            }
            
            # Store data in database
            await self._store_sensor_data(system_id, processed_data, anomalies)
            
            # Handle any detected anomalies
            if anomalies["detected"]:
                await self._handle_anomalies(system_id, anomalies)
            
            # Prepare response
            response = {
                "timestamp": datetime.now().isoformat(),
                "system_id": system_id,
                "data": processed_data,
                "anomalies": anomalies,
                "status": "processed"
            }
            
            # Broadcast update to connected clients
            await self.connection_manager.broadcast_to_system(
                system_id,
                response
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {str(e)}")
            raise RealTimeProcessingError(f"Data processing failed: {str(e)}")

    def _preprocess_sensor_data(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Preprocess and validate sensor data."""
        required_fields = [
            "temperature", "humidity", "pressure",
            "power", "flow_rate"
        ]
        
        # Validate required fields
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise RealTimeProcessingError(f"Missing required fields: {missing}")
        
        # Convert to float and handle missing values
        processed = {}
        for key, value in data.items():
            try:
                processed[key] = float(value) if value is not None else None
            except (ValueError, TypeError):
                processed[key] = None
        
        return processed

    async def _detect_anomalies(
        self,
        system_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect anomalies in sensor data."""
        try:
            # Prepare data for autoencoder
            input_data = np.array([list(data.values())])
            
            # Detect anomalies
            anomalies, errors = self.autoencoder.detect_anomalies(
                input_data,
                threshold_multiplier=self.alert_config.threshold_multiplier
            )
            
            # Determine alert level
            alert_level = self._determine_alert_level(
                errors[0],
                self.alert_config.alert_levels
            )
            
            # Update anomaly counter
            if anomalies[0]:
                self.anomaly_counters[system_id] = self.anomaly_counters.get(system_id, 0) + 1
            else:
                self.anomaly_counters[system_id] = 0
            
            return {
                "detected": bool(anomalies[0]),
                "error_score": float(errors[0]),
                "alert_level": alert_level,
                "consecutive_count": self.anomaly_counters[system_id]
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return {
                "detected": False,
                "error_score": 0.0,
                "alert_level": "unknown",
                "consecutive_count": 0
            }

    def _determine_alert_level(
        self,
        error_score: float,
        alert_levels: Dict[str, float]
    ) -> str:
        """Determine alert level based on error score."""
        for level, threshold in sorted(
            alert_levels.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if error_score >= threshold:
                return level
        return "normal"

    async def _handle_anomalies(
        self,
        system_id: str,
        anomaly_data: Dict[str, Any]
    ):
        """Handle detected anomalies."""
        # Check if we should trigger an alert
        should_alert = (
            anomaly_data["consecutive_count"] >= 
            self.alert_config.min_consecutive_anomalies
        )
        
        if should_alert:
            # Check cooldown period
            last_alert = self.last_alert_time.get(system_id)
            if not last_alert or (
                datetime.now() - last_alert
            ).total_seconds() >= self.alert_config.cooldown_period:
                await self._trigger_alert(system_id, anomaly_data)
                self.last_alert_time[system_id] = datetime.now()

    async def _trigger_alert(
        self,
        system_id: str,
        anomaly_data: Dict[str, Any]
    ):
        """Trigger alert for anomaly."""
        alert = {
            "type": "anomaly_alert",
            "system_id": system_id,
            "timestamp": datetime.now().isoformat(),
            "alert_level": anomaly_data["alert_level"],
            "error_score": anomaly_data["error_score"],
            "consecutive_anomalies": anomaly_data["consecutive_count"]
        }
        
        # Broadcast alert to connected clients
        await self.connection_manager.broadcast_to_system(
            system_id,
            alert
        )
        
        # Store alert in database
        await self.db.save_anomaly_event({
            "system_id": system_id,
            "timestamp": datetime.now(),
            "alert_level": anomaly_data["alert_level"],
            "error_score": anomaly_data["error_score"],
            "details": anomaly_data
        })

    async def _store_sensor_data(
        self,
        system_id: str,
        data: Dict[str, Any],
        anomalies: Dict[str, Any]
    ):
        """Store sensor data and anomaly information."""
        try:
            # Store raw sensor data
            await self.db.save_system_status({
                "system_id": system_id,
                "timestamp": datetime.now(),
                "data": data,
                "anomalies": anomalies
            })
            
        except Exception as e:
            logger.error(f"Failed to store sensor data: {str(e)}")

    async def handle_websocket(
        self,
        websocket: WebSocket,
        client_id: str,
        system_id: str
    ):
        """Handle WebSocket connection for real-time updates."""
        try:
            await self.connection_manager.connect(
                websocket,
                client_id,
                system_id
            )
            
            # Send initial system state if available
            if system_id in self.system_states:
                await websocket.send_json({
                    "type": "initial_state",
                    "data": self.system_states[system_id]
                })
            
            # Handle incoming messages
            while True:
                try:
                    message = await websocket.receive_json()
                    await self.process_client_message(websocket, message)
                except WebSocketDisconnect:
                    self.connection_manager.disconnect(websocket)
                    break
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                    
        except Exception as e:
            logger.error(f"WebSocket handler error: {str(e)}")
            if websocket in self.connection_manager.client_info:
                self.connection_manager.disconnect(websocket)

    async def process_client_message(
        self,
        websocket: WebSocket,
        message: Dict[str, Any]
    ):
        """Process message from connected client."""
        message_type = message.get("type")
        
        if message_type == "subscribe":
            # Handle subscription request
            pass
        elif message_type == "unsubscribe":
            # Handle unsubscribe request
            pass
        elif message_type == "command":
            # Handle command request
            pass
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            })

    async def start_processing(self, system_id: str):
        """Start processing for a system."""
        if system_id in self.processing_tasks:
            return
            
        self.processing_tasks[system_id] = asyncio.create_task(
            self._processing_loop(system_id)
        )

    async def stop_processing(self, system_id: str):
        """Stop processing for a system."""
        if system_id in self.processing_tasks:
            self.processing_tasks[system_id].cancel()
            del self.processing_tasks[system_id]

    async def _processing_loop(self, system_id: str):
        """Main processing loop for a system."""
        try:
            while True:
                # Process any pending data
                if system_id in self.system_states:
                    current_state = self.system_states[system_id]
                    if (datetime.now() - current_state["last_update"]).seconds > 60:
                        # Alert if no recent updates
                        await self._trigger_alert(
                            system_id,
                            {
                                "alert_level": "warning",
                                "message": "No recent data updates"
                            }
                        )
                
                await asyncio.sleep(1)  # Processing interval
                
        except asyncio.CancelledError:
            logger.info(f"Processing loop cancelled for system {system_id}")
        except Exception as e:
            logger.error(f"Processing loop error for system {system_id}: {str(e)}")

    async def close(self):
        """Close all connections and cleanup."""
        try:
            # Cancel all processing tasks
            for task in self.processing_tasks.values():
                task.cancel()
            
            # Close database connection
            self.db.close()
            
            # Close all WebSocket connections
            for connections in self.connection_manager.active_connections.values():
                for websocket in connections:
                    await websocket.close()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
