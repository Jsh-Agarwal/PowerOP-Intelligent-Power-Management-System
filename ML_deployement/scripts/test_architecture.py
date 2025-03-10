import numpy as np
from datetime import datetime, timedelta
import asyncio
from models.lstm_model import LSTMModel
from models.autoencoder import Autoencoder
from services.groq_slm_service import GroqSLMService

async def test_architecture_integration():
    print("Testing HVAC Neural Engine Architecture Integration")
    print("------------------------------------------------")

    # 1. Generate sample data
    print("\n1. Generating sample data...")
    sample_data = {
        "temperature": np.random.normal(22, 2, 24),
        "humidity": np.random.normal(50, 5, 24),
        "power": np.random.normal(1000, 100, 24),
        "occupancy": np.random.randint(0, 100, 24),
    }
    
    # 2. Test LSTM Prediction
    print("\n2. Testing LSTM Temperature Prediction...")
    lstm_model = LSTMModel(input_shape=(24, len(sample_data)))
    try:
        predictions = await lstm_model.predict_next_24h(
            current_data=sample_data,
            weather_forecast=None  # No weather data for test
        )
        print(f"✓ LSTM Prediction Shape: {len(predictions['predictions'])} hours")
    except Exception as e:
        print(f"✗ LSTM Error: {str(e)}")

    # 3. Test Autoencoder Anomaly Detection
    print("\n3. Testing Autoencoder Anomaly Detection...")
    autoencoder = Autoencoder(input_dim=len(sample_data))
    try:
        # Create normal and anomalous data
        normal_data = np.column_stack([v for v in sample_data.values()])
        anomalous_data = normal_data.copy()
        anomalous_data[0] += 10  # Create obvious anomaly
        
        # Test detection
        anomalies, scores = autoencoder.detect_anomalies(
            np.vstack([normal_data, anomalous_data])
        )
        print(f"✓ Autoencoder detected {sum(anomalies)} anomalies")
    except Exception as e:
        print(f"✗ Autoencoder Error: {str(e)}")

    # 4. Test Groq SLM Integration
    print("\n4. Testing Groq SLM Optimization...")
    groq_service = GroqSLMService()
    try:
        # Prepare system state
        system_state = {
            "current_temperature": 23.5,
            "target_temperature": 22.0,
            "humidity": 55,
            "power_consumption": 1200,
            "predictions": predictions["predictions"],
            "anomalies_detected": sum(anomalies),
            "anomaly_scores": scores.tolist()
        }
        
        # Get optimization recommendations
        optimization = await groq_service.generate_hvac_optimization(
            hvac_data=system_state,
            optimization_target="efficiency"
        )
        
        print("\nOptimization Results:")
        print(f"✓ Recommendations: {len(optimization['recommendations'])}")
        print(f"✓ Expected Savings: {optimization['expected_savings']}%")
        print(f"✓ Confidence Score: {optimization['confidence_score']}")
        
    except Exception as e:
        print(f"✗ Groq SLM Error: {str(e)}")
    finally:
        await groq_service.close()

    print("\n------------------------------------------------")
    print("Architecture Integration Test Complete")

if __name__ == "__main__":
    asyncio.run(test_architecture_integration())
