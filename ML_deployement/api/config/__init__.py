"""
Configuration management for PowerOP_ML
"""
from pathlib import Path
import os

# Base directories
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / 'config'
MODELS_DIR = BASE_DIR / 'models'

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
