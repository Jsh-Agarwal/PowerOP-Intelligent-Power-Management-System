from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging
from .lstm_model import LSTMModel
from .autoencoder import AutoEncoderModel

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent.parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
    def load_lstm_model(self, input_shape: tuple) -> LSTMModel:
        """Load LSTM model with specified input shape."""
        return LSTMModel(input_shape=input_shape)
        
    def load_autoencoder_model(self) -> AutoEncoderModel:
        """Load AutoEncoder model for anomaly detection."""
        return AutoEncoderModel()
        
    def save_model_metadata(
        self,
        model_name: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Save model metadata."""
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
