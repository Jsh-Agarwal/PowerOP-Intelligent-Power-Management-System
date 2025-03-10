import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .exceptions import DataProcessingError
from .logger import setup_logger

logger = setup_logger('utilities')

# Data Preprocessing Functions
def load_hvac_data(filepath: str) -> pd.DataFrame:
    """Load HVAC data from CSV file."""
    try:
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        raise DataProcessingError(f"Failed to load data: {str(e)}")

def normalize_features(
    data: pd.DataFrame,
    columns: List[str],
    scaler_type: str = 'standard'
) -> Tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]:
    """Normalize selected features using specified scaler."""
    try:
        scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        data_scaled = data.copy()
        data_scaled[columns] = scaler.fit_transform(data[columns])
        return data_scaled, scaler
    except Exception as e:
        raise DataProcessingError(f"Normalization failed: {str(e)}")

def create_rolling_features(
    data: pd.DataFrame,
    columns: List[str],
    window: int = 24
) -> pd.DataFrame:
    """Create rolling mean features for specified columns."""
    df = data.copy()
    for col in columns:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window).mean()
    return df

# Statistical Functions
def calculate_anomaly_thresholds(
    series: pd.Series,
    n_std: float = 3
) -> Tuple[float, float]:
    """Calculate anomaly thresholds using standard deviation."""
    mean = series.mean()
    std = series.std()
    return mean - n_std * std, mean + n_std * std

def calculate_confidence_interval(
    series: pd.Series,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval for a series."""
    mean = series.mean()
    std_err = series.std() / np.sqrt(len(series))
    z_score = abs(np.percentile(np.random.normal(0, 1, 10000), 
                               (1 - confidence) * 100 / 2))
    margin = z_score * std_err
    return mean - margin, mean + margin

def detect_seasonal_pattern(
    series: pd.Series,
    period: int = 24
) -> np.ndarray:
    """Detect seasonal pattern in time series data."""
    seasonal = series.values.reshape(-1, period).mean(axis=0)
    return seasonal

# Feature Engineering Functions
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from datetime index."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    return df

def calculate_system_efficiency(
    df: pd.DataFrame,
    power_col: str = 'active_power',
    output_col: str = 'active_energy'
) -> pd.Series:
    """Calculate system efficiency metrics."""
    return (df[output_col] / df[power_col]).replace([np.inf, -np.inf], np.nan)

def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int]
) -> pd.DataFrame:
    """Create lagged features for specified columns."""
    df_new = df.copy()
    for col in columns:
        for lag in lags:
            df_new[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df_new

# Additional Data Validation Functions
def validate_hvac_data(df: pd.DataFrame) -> bool:
    """Validate HVAC data contains required columns."""
    required_columns = [
        'on_off', 'damper', 'active_energy', 'co2_1', 'amb_humid_1',
        'active_power', 'pot_gen', 'high_pressure_1', 'high_pressure_2',
        'low_pressure_1', 'low_pressure_2', 'high_pressure_3', 'low_pressure_3',
        'outside_temp', 'outlet_temp', 'inlet_temp', 'summer_setpoint_temp',
        'winter_setpoint_temp', 'amb_temp_2'
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise DataProcessingError(f"Missing required columns: {missing}")
    return True

def clean_hvac_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean HVAC data by handling missing values and outliers."""
    df_clean = df.copy()
    
    # Handle missing values
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_columns] = df_clean[numeric_columns].fillna(
        df_clean[numeric_columns].rolling(window=24, min_periods=1).mean()
    )
    
    # Remove outliers using IQR method
    for col in numeric_columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
        df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
    
    return df_clean

def calculate_hvac_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional HVAC performance metrics."""
    df_metrics = df.copy()
    
    # Calculate temperature differential
    df_metrics['temp_differential'] = df_metrics['outlet_temp'] - df_metrics['inlet_temp']
    
    # Calculate pressure means
    df_metrics['high_pressure_mean'] = df_metrics[
        ['high_pressure_1', 'high_pressure_2', 'high_pressure_3']
    ].mean(axis=1)
    
    df_metrics['low_pressure_mean'] = df_metrics[
        ['low_pressure_1', 'low_pressure_2', 'low_pressure_3']
    ].mean(axis=1)
    
    # Calculate system states
    df_metrics['is_cooling'] = (
        (df_metrics['outlet_temp'] < df_metrics['inlet_temp']) & 
        (df_metrics['on_off'] == 1)
    ).astype(int)
    
    df_metrics['is_heating'] = (
        (df_metrics['outlet_temp'] > df_metrics['inlet_temp']) & 
        (df_metrics['on_off'] == 1)
    ).astype(int)
    
    # Calculate load factor
    max_power = df_metrics['active_power'].max()
    df_metrics['load_factor'] = df_metrics['active_power'] / max_power
    
    return df_metrics

def prepare_model_features(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    lookback: int = 24
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for model training."""
    if feature_cols is None:
        feature_cols = [
            'outside_temp', 'inlet_temp', 'outlet_temp', 'active_power',
            'high_pressure_mean', 'low_pressure_mean', 'load_factor'
        ]
    
    # Create lagged features
    df_features = create_lag_features(df[feature_cols], feature_cols, [1, 2, 3])
    
    # Create rolling means
    df_features = create_rolling_features(df_features, feature_cols, lookback)
    
    # Add time features
    df_features = create_time_features(df_features)
    
    # Drop rows with NaN values
    df_features = df_features.dropna()
    
    # Separate target
    y = df_features[target_col]
    X = df_features.drop(target_col, axis=1)
    
    return X, y

def process_hvac_data_pipeline(
    filepath: str,
    target_col: str = 'active_energy'
) -> Tuple[pd.DataFrame, pd.Series]:
    """Complete data processing pipeline for HVAC data."""
    # Load data
    df = load_hvac_data(filepath)
    
    # Validate data
    validate_hvac_data(df)
    
    # Clean data
    df_clean = clean_hvac_data(df)
    
    # Calculate metrics
    df_processed = calculate_hvac_metrics(df_clean)
    
    # Prepare features for modeling
    X, y = prepare_model_features(df_processed, target_col)
    
    return X, y
