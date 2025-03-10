import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from models.lstm_model import LSTMModel

def generate_synthetic_data(days: int = 30):
    """Generate synthetic training data."""
    timestamps = [
        datetime.now() - timedelta(days=i)
        for i in range(days)
    ]
    
    # Generate synthetic temperature data with daily patterns
    temperatures = [
        22 + 5 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 1)
        for i in range(days * 24)
    ]
    
    # Generate related humidity data
    humidity = [
        50 + 20 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 5)
        for i in range(days * 24)
    ]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures,
        'humidity': humidity
    })

def train_and_save_model():
    """Train initial model and save it."""
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Generate training data
    data = generate_synthetic_data()
    
    # Prepare sequences
    sequence_length = 24
    features = ['temperature', 'humidity']
    
    # Initialize and train model
    model = LSTMModel(input_shape=(sequence_length, len(features)))
    X, y = model.preprocess_data(data, features, 'temperature')
    
    # Train model
    model.train(
        X_train=X,
        y_train=y,
        epochs=10,
        batch_size=32,
        model_path=str(models_dir / "temperature_lstm.h5")
    )
    
    print("Initial model trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
