import os
from typing import Any, Dict
import yaml
from .exceptions import ConfigurationError

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to load config: {str(e)}")

def get_env_var(name: str, default: Any = None) -> Any:
    """Get environment variable with optional default."""
    return os.environ.get(name, default)

def validate_config(config: Dict[str, Any], required_fields: list) -> bool:
    """Validate configuration contains required fields."""
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ConfigurationError(f"Missing required config fields: {missing}")
    return True

def get_hvac_config_defaults() -> Dict[str, Any]:
    """Get default configuration for HVAC system."""
    return {
        'data_path': 'data/HVAC_processed.csv',
        'model_path': 'models',
        'log_path': 'logs',
        'feature_columns': [
            'outside_temp', 'inlet_temp', 'outlet_temp', 'active_power',
            'high_pressure_mean', 'low_pressure_mean', 'load_factor'
        ],
        'target_column': 'active_energy',
        'lookback_window': 24,
        'train_test_split': 0.2,
        'random_state': 42
    }

def validate_hvac_config(config: Dict[str, Any]) -> bool:
    """Validate HVAC-specific configuration."""
    required_fields = [
        'data_path',
        'model_path',
        'log_path',
        'feature_columns',
        'target_column'
    ]
    
    # Check required fields
    validate_config(config, required_fields)
    
    # Validate paths exist
    data_dir = os.path.dirname(config['data_path'])
    if not os.path.exists(data_dir):
        raise ConfigurationError(f"Data directory does not exist: {data_dir}")
    
    # Create directories if they don't exist
    os.makedirs(config['model_path'], exist_ok=True)
    os.makedirs(config['log_path'], exist_ok=True)
    
    return True

def load_hvac_config(config_path: str) -> Dict[str, Any]:
    """Load and validate HVAC configuration."""
    config = load_config(config_path)
    defaults = get_hvac_config_defaults()
    
    # Merge with defaults
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    # Validate configuration
    validate_hvac_config(config)
    
    return config
