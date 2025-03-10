class ModelManager:
    """Manager for ML models."""
    
    def __init__(self):
        self.model_registry = {}
        
    def load_lstm_model(self, input_shape):
        """Load LSTM model with given input shape."""
        from services.models.lstm_model import LSTMModel
        model = LSTMModel(input_shape=input_shape)
        return model
        
    def save_lstm_model(self, model, name, metadata=None):
        """Save LSTM model to registry."""
        model_path = f"/models/{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.model"
        # In a real implementation, save the model to disk or database
        self.model_registry[name] = {
            "path": model_path,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        return model_path
