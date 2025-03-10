import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import asyncio

from models.lstm_model import LSTMModel
from models.autoencoder import Autoencoder

def generate_training_data(days: int = 30):
    """Generate synthetic data for training."""
    timestamps = [
        datetime.now() - timedelta(hours=i)
        for i in range(days * 24)
    ]
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': [
            22 + 5 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 1)
            for i in range(days * 24)
        ],
        'humidity': [
            50 + 20 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 5)
            for i in range(days * 24)
        ],
        'power': [
            1000 + 200 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 50)
            for i in range(days * 24)
        ]
    })
    return data

async def train_models():
    """Train and save both LSTM and Autoencoder models."""
    print("Starting model training...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Generate training data
    data = generate_training_data()
    print("Generated synthetic training data")
    
    try:
        # Train LSTM model
        print("\nTraining LSTM model...")
        lstm = LSTMModel(input_shape=(24, 3))  # 3 features
        features = ['temperature', 'humidity', 'power']
        X, y = lstm.preprocess_data(data, features, 'temperature')
        
        history = lstm.train(
            X_train=X,
            y_train=y,
            epochs=10,
            batch_size=32
        )
        
        # Save LSTM model
        lstm_path = models_dir / "temperature_lstm.h5"
        lstm.save_model(str(lstm_path))
        print(f"LSTM model saved to {lstm_path}")
        
        # Train Autoencoder
        print("\nTraining Autoencoder model...")
        autoencoder = Autoencoder(input_dim=len(features))
        normalized_data = (data[features] - data[features].mean()) / data[features].std()
        
        autoencoder.train(
            normalized_data.values,
            epochs=10,
            batch_size=32
        )
        
        # Save Autoencoder model
        autoencoder_path = models_dir / "anomaly_autoencoder.h5"
        autoencoder.save_model(str(autoencoder_path))
        print(f"Autoencoder model saved to {autoencoder_path}")
        
        print("\nModel training completed successfully!")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")

if __name__ == "__main__":
    asyncio.run(train_models())
