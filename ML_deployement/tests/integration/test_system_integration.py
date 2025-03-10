import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta

@pytest.mark.integration
class TestSystemIntegration:
    """System-wide integration tests."""
    
    @pytest.mark.asyncio
    async def test_temperature_prediction_workflow(
        self,
        test_lstm_model,
        mock_weather_service,
        mock_astra_service,
        synthetic_hvac_data
    ):
        """Test complete temperature prediction workflow."""
        # Setup initial conditions
        current_data = synthetic_hvac_data
        weather_forecast = await mock_weather_service.get_forecast("test_location")
        
        # Test prediction generation
        predictions = await test_lstm_model.predict_next_24h(
            current_data,
            weather_forecast
        )
        
        # Verify predictions
        assert len(predictions["predictions"]) == 24
        assert all(18.0 <= temp <= 28.0 for temp in predictions["predictions"])
        
        # Test data storage
        mock_astra_service.save_temperature_data.assert_called_once()
        stored_data = mock_astra_service.save_temperature_data.call_args[0][0]
        assert "temperature" in stored_data
        assert "predictions" in stored_data

    @pytest.mark.asyncio
    async def test_optimization_workflow(
        self,
        comfort_optimizer,
        energy_optimizer,
        synthetic_hvac_data,
        test_system_config
    ):
        """Test complete optimization workflow."""
        # Test comfort optimization
        comfort_result = await comfort_optimizer.optimize_comfort(
            current_state=synthetic_hvac_data,
            constraints=test_system_config
        )
        
        assert "recommendations" in comfort_result
        assert comfort_result["comfort_score"] >= test_system_config["comfort_threshold"]
        
        # Test energy optimization
        energy_result = await energy_optimizer.optimize_energy_cost(
            current_state=synthetic_hvac_data,
            constraints=test_system_config
        )
        
        assert "recommendations" in energy_result
        assert energy_result["expected_savings"] > 0
        
        # Verify recommendations don't conflict
        comfort_temps = [r["value"] for r in comfort_result["recommendations"] 
                        if r["type"] == "temperature"]
        energy_temps = [r["value"] for r in energy_result["recommendations"]
                       if r["type"] == "temperature"]
        
        # Check temperature recommendations are within acceptable range
        for temp in comfort_temps + energy_temps:
            assert test_system_config["temperature_range"][0] <= temp <= \
                   test_system_config["temperature_range"][1]

    @pytest.mark.asyncio
    async def test_anomaly_detection_workflow(
        self,
        test_autoencoder,
        real_time_processor,
        synthetic_hvac_data
    ):
        """Test anomaly detection and alert workflow."""
        # Generate anomalous data
        anomalous_data = synthetic_hvac_data.copy()
        anomalous_data["temperature"] += 10.0  # Significant deviation
        
        # Test anomaly detection
        result = await real_time_processor.process_sensor_data(
            "test_system",
            anomalous_data
        )
        
        assert result["anomalies"]["detected"] == True
        assert result["anomalies"]["alert_level"] in ["warning", "critical"]
        
        # Verify alert generation
        assert len(real_time_processor.connection_manager.broadcast_messages) > 0
        alert = real_time_processor.connection_manager.broadcast_messages[-1]
        assert alert["type"] == "anomaly_alert"
        assert alert["system_id"] == "test_system"

    @pytest.mark.asyncio
    async def test_data_persistence_workflow(
        self,
        mock_astra_service,
        synthetic_hvac_data,
        test_system_config
    ):
        """Test data persistence across components."""
        # Test temperature data storage
        temp_data = {
            **synthetic_hvac_data,
            "system_id": "test_system",
            "timestamp": datetime.now()
        }
        
        await mock_astra_service.save_temperature_data(temp_data)
        
        # Test optimization result storage
        opt_result = {
            "system_id": "test_system",
            "timestamp": datetime.now(),
            "recommendations": [
                {"type": "temperature", "value": 22.0}
            ],
            "expected_savings": 15.0
        }
        
        await mock_astra_service.save_optimization_result(opt_result)
        
        # Verify data retrieval
        stored_temp = await mock_astra_service.get_temperature_data(
            system_id="test_system",
            limit=1
        )
        assert stored_temp[0]["temperature"] == temp_data["temperature"]
        
        stored_opt = await mock_astra_service.get_optimization_history(
            system_id="test_system",
            limit=1
        )
        assert stored_opt[0]["expected_savings"] == opt_result["expected_savings"]

    @pytest.mark.asyncio
    async def test_real_time_processing_workflow(
        self,
        real_time_processor,
        test_autoencoder,
        synthetic_hvac_data
    ):
        """Test real-time data processing workflow."""
        # Start processing
        await real_time_processor.start_processing("test_system")
        
        # Send test data
        await real_time_processor.process_sensor_data(
            "test_system",
            synthetic_hvac_data
        )
        
        # Verify processing
        assert "test_system" in real_time_processor.system_states
        state = real_time_processor.system_states["test_system"]
        assert "current_data" in state
        assert "anomalies" in state
        
        # Test WebSocket updates
        assert len(real_time_processor.connection_manager.broadcast_messages) > 0
        
        # Stop processing
        await real_time_processor.stop_processing("test_system")
