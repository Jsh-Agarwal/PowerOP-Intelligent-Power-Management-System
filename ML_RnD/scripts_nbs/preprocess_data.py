import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from datetime import datetime, timedelta
import holidays
from typing import Tuple, List, Dict, Union
import logging

class HVACDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for HVAC system data.
    Handles missing values, feature engineering, and data validation.
    """
    
    def __init__(self, 
                 scaler_type: str = 'standard',
                 imputer_n_neighbors: int = 5,
                 country_holidays: str = 'US'):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
            imputer_n_neighbors: Number of neighbors for KNN imputation
            country_holidays: Country code for holiday detection
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors=imputer_n_neighbors)
        self.holidays = holidays.CountryHoliday(country_holidays)
        self.feature_names = None
        self.numerical_columns = None
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the preprocessor"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _validate_raw_data(self, df: pd.DataFrame) -> None:
        """
        Validate the input data structure and content.
        
        Args:
            df: Input DataFrame
        Raises:
            ValueError: If data validation fails
        """
        required_columns = [
            'Date', 'on_off', 'damper', 'active_energy', 'co2_1', 'amb_humid_1',
            'active_power', 'pot_gen', 'high_pressure_1', 'high_pressure_2',
            'low_pressure_1', 'low_pressure_2', 'high_pressure_3', 'low_pressure_3',
            'outside_temp', 'outlet_temp', 'inlet_temp', 'summer_setpoint_temp',
            'winter_setpoint_temp', 'amb_temp_2'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Check data types
        try:
            pd.to_datetime(df['Date'])
        except:
            raise ValueError("Date column cannot be parsed as datetime")

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies.
        
        Args:
            df: Input DataFrame
        Returns:
            DataFrame with handled missing values
        """
        # Fill boolean columns
        boolean_columns = ['on_off', 'damper']
        df[boolean_columns] = df[boolean_columns].fillna(0)
        
        # Handle numerical columns with KNN imputation
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        df[numerical_columns] = self.imputer.fit_transform(df[numerical_columns])
        
        return df

    def _engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from the Date column.
        
        Args:
            df: Input DataFrame
        Returns:
            DataFrame with additional time-based features
        """
        df['datetime'] = pd.to_datetime(df['Date'])
        
        # Extract basic time components
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Create cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Add holiday indicator
        df['is_holiday'] = df['datetime'].apply(lambda x: x in self.holidays).astype(int)
        
        return df

    def _engineer_hvac_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create HVAC-specific engineered features.
        
        Args:
            df: Input DataFrame
        Returns:
            DataFrame with additional HVAC-specific features
        """
        # Temperature differences
        df['temp_difference_in_out'] = df['outlet_temp'] - df['inlet_temp']
        df['temp_difference_ambient'] = df['outside_temp'] - df['inlet_temp']
        
        # Pressure ratios and differences
        df['high_pressure_avg'] = df[['high_pressure_1', 'high_pressure_2', 'high_pressure_3']].mean(axis=1)
        df['low_pressure_avg'] = df[['low_pressure_1', 'low_pressure_2', 'low_pressure_3']].mean(axis=1)
        df['pressure_ratio'] = df['high_pressure_avg'] / df['low_pressure_avg']
        
        # System efficiency indicators
        df['power_per_temp_diff'] = df['active_power'] / (df['temp_difference_in_out'] + 1e-6)
        df['energy_efficiency'] = df['active_energy'] / (df['active_power'] + 1e-6)
        
        # Comfort indicators
        df['temp_setpoint_diff'] = np.where(
            df['month'].isin([6, 7, 8]),  # Summer months
            df['inlet_temp'] - df['summer_setpoint_temp'],
            df['inlet_temp'] - df['winter_setpoint_temp']
        )
        
        return df

    def _create_rolling_features(self, df: pd.DataFrame, windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """
        Create rolling window features for key metrics.
        
        Args:
            df: Input DataFrame
            windows: List of window sizes (in hours) for rolling calculations
        Returns:
            DataFrame with additional rolling features
        """
        key_metrics = ['active_power', 'inlet_temp', 'co2_1', 'amb_humid_1']
        
        for window in windows:
            for metric in key_metrics:
                # Rolling mean
                df[f'{metric}_rolling_mean_{window}h'] = (
                    df[metric].rolling(window=window * 12, min_periods=1).mean()
                )
                # Rolling std
                df[f'{metric}_rolling_std_{window}h'] = (
                    df[metric].rolling(window=window * 12, min_periods=1).std()
                )
        
        return df

    def _prepare_target_variable(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the target variable (active_power) for modeling.
        
        Args:
            df: Input DataFrame
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        target = df['active_power']
        features = df.drop(['active_power', 'datetime', 'Date'], axis=1)
        
        return features, target

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using the configured scaler.
        
        Args:
            df: Input DataFrame
        Returns:
            DataFrame with scaled features
        """
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        return df

    def preprocess(self, 
                  df: pd.DataFrame, 
                  training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            training: Whether this is for training data (True) or inference (False)
        Returns:
            Tuple of (preprocessed features, target variable)
        """
        self.logger.info("Starting preprocessing pipeline...")
        
        # Validate raw data
        self._validate_raw_data(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Engineer features
        df = self._engineer_time_features(df)
        df = self._engineer_hvac_features(df)
        df = self._create_rolling_features(df)
        
        # Prepare features and target
        features, target = self._prepare_target_variable(df)
        
        # Scale features
        features = self._scale_features(features)
        
        if training:
            self.feature_names = features.columns.tolist()
            self.numerical_columns = features.select_dtypes(include=[np.number]).columns.tolist()
        
        self.logger.info("Preprocessing pipeline completed successfully.")
        return features, target

    def get_feature_names(self) -> List[str]:
        """Return the list of feature names after preprocessing"""
        if self.feature_names is None:
            raise ValueError("Preprocessor hasn't been fitted with training data yet")
        return self.feature_names

class DataValidator:
    """
    Validates data quality and generates reports on data issues.
    """
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Union[float, List[str]]]:
        """
        Perform comprehensive data quality checks.
        
        Args:
            df: Input DataFrame
        Returns:
            Dictionary containing quality metrics and issues
        """
        quality_report = {
            'missing_values': {},
            'outliers': {},
            'inconsistencies': [],
            'data_coverage': {}
        }
        
        # Check missing values
        missing_vals = df.isnull().sum()
        quality_report['missing_values'] = {
            col: count for col, count in missing_vals.items() if count > 0
        }
        
        # Check value ranges
        for col in df.select_dtypes(include=[np.number]).columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[
                (df[col] < (q1 - 1.5 * iqr)) | 
                (df[col] > (q3 + 1.5 * iqr))
            ]
            if len(outliers) > 0:
                quality_report['outliers'][col] = len(outliers)
        
        # Check data consistency
        if 'datetime' in df.columns:
            time_gaps = df['datetime'].diff().dt.total_seconds() / 60
            irregular_intervals = time_gaps[time_gaps != 5].index
            if len(irregular_intervals) > 0:
                quality_report['inconsistencies'].append(
                    f"Irregular time intervals found at {len(irregular_intervals)} points"
                )
        
        # Check data coverage
        if 'datetime' in df.columns:
            date_range = df['datetime'].max() - df['datetime'].min()
            expected_records = (date_range.total_seconds() / 300) + 1  # 5-minute intervals
            coverage = len(df) / expected_records * 100
            quality_report['data_coverage'] = {
                'start_date': df['datetime'].min(),
                'end_date': df['datetime'].max(),
                'coverage_percentage': coverage
            }
        
        return quality_report

def prepare_sequence_data(features: pd.DataFrame, 
                         target: pd.Series,
                         sequence_length: int = 24,
                         forecast_horizon: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequential data for time series models.
    
    Args:
        features: Preprocessed feature DataFrame
        target: Target variable Series
        sequence_length: Number of time steps in each sequence
        forecast_horizon: Number of steps to forecast
    Returns:
        Tuple of (X sequences, y sequences)
    """
    X, y = [], []
    
    for i in range(len(features) - sequence_length - forecast_horizon + 1):
        X.append(features.iloc[i:(i + sequence_length)].values)
        y.append(target.iloc[i + sequence_length:i + sequence_length + forecast_horizon].values)
    
    return np.array(X), np.array(y)

# Example usage:
def process_hvac_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process HVAC data from raw file to model-ready format.
    
    Args:
        data_path: Path to raw data file
    Returns:
        Tuple of (processed features, target variable)
    """
    # Read data
    df = pd.read_csv(data_path, parse_dates=['Date'])
    
    # Initialize preprocessor
    preprocessor = HVACDataPreprocessor(
        scaler_type='standard',
        imputer_n_neighbors=5
    )
    
    # Validate data quality
    quality_report = DataValidator.check_data_quality(df)
    logging.info(f"Data quality report: {quality_report}")
    
    # Preprocess data
    features, target = preprocessor.preprocess(df, training=True)
    
    # Prepare sequential data
    X_seq, y_seq = prepare_sequence_data(
        features,
        target,
        sequence_length=24,  # 2 hours of 5-minute data
        forecast_horizon=12  # Predict next hour
    )
    
    return X_seq, y_seq