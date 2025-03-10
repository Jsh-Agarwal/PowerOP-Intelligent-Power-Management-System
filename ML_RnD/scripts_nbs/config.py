from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path("D:/PowerAmp")
    STORAGE_DIR = BASE_DIR / "storage"
    MODEL_DIR = STORAGE_DIR / "models"
    RESULTS_DIR = STORAGE_DIR / "results"
    
    # Model settings
    RETRAIN_IF_EXISTS = False
    FORCE_RETRAIN = False
    
    # Model filenames
    MODEL_FILES = {
        'bi_lstm': 'bi_lstm.keras',
        'attention': 'attention.keras',
        'combined': 'combined.keras'
    }
    
    # Model parameters
    MODEL_PARAMS = {
        'sequence_length': 24,
        'forecast_horizon': 12,
        'batch_size': 64,
        'epochs': 50
    }
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Create instance for import
config = Config()
