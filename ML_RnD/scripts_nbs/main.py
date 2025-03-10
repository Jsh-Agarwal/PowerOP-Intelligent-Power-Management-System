import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# Import custom modules
from preprocess_data import HVACDataPreprocessor, DataValidator, process_hvac_data
from regression import ModelTrainer, ModelEvaluator
from anomaly import AnomalyDetectorEnsemble
from shap import ExplainableAI, analyze_model_explanability

class HVACAnalysisPipeline:
    """
    Main pipeline class that coordinates data processing, model training,
    and result generation for HVAC system analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.setup_directories()
        self.setup_logging()
        
    def setup_directories(self) -> None:
        """Create necessary directories for outputs"""
        # Create timestamp-based output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.config['output_base_dir']) / f"analysis_{timestamp}"
        
        # Create subdirectories
        self.subdirs = {
            'models': self.output_dir / 'models',
            'plots': self.output_dir / 'plots',
            'results': self.output_dir / 'results',
            'anomalies': self.output_dir / 'anomalies',
            'shap': self.output_dir / 'shap'
        }
        
        for dir_path in self.subdirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self) -> None:
        """Configure logging for the pipeline"""
        log_file = self.output_dir / 'pipeline.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file and perform initial validation.
        
        Returns:
            DataFrame containing raw data
        """
        self.logger.info(f"Loading data from {self.config['data_path']}")
        try:
            df = pd.read_csv(self.config['data_path'])
            self.logger.info(f"Loaded {len(df)} records")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save results to JSON file"""
        file_path = self.subdirs['results'] / filename
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
            
    def plot_model_performance(self, results: Dict[str, Any]) -> None:
        """Generate and save model performance plots"""
        # Training history plot
        plt.figure(figsize=(12, 6))
        for model_name, history in results['training_history'].items():
            plt.plot(history.history['loss'], label=f'{model_name}_train')
            plt.plot(history.history['val_loss'], label=f'{model_name}_val')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.subdirs['plots'] / 'training_history.png')
        plt.close()
        
        # Model comparison plot
        plt.figure(figsize=(10, 6))
        metrics = pd.DataFrame(results['evaluation_results']).T
        metrics['r2'].plot(kind='bar')
        plt.title('Model R² Scores')
        plt.tight_layout()
        plt.savefig(self.subdirs['plots'] / 'model_comparison.png')
        plt.close()
        
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete analysis pipeline.
        
        Returns:
            Dictionary containing all results and metrics
        """
        try:
            # 1. Load and preprocess data
            self.logger.info("Starting data preprocessing")
            raw_data = self.load_data()
            
            # Validate data quality
            quality_report = DataValidator.check_data_quality(raw_data)
            self.save_results(quality_report, 'data_quality_report.json')
            
            # Process data
            X_processed, y = process_hvac_data(self.config['data_path'])
            
            # 2. Split data
            split_idx = int(len(X_processed) * 0.8)
            X_train, X_test = X_processed[:split_idx], X_processed[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Preprocess data
            X_train, y_train = preprocess_data(X_train, y_train)
            
            # 3. Train regression models
            self.logger.info("Training regression models")
            trainer = ModelTrainer(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                output_dim=y_train.shape[1]  # Add output dimension
            )
            trainer.compile_models()
            training_results = trainer.train_all_models(X_train, y_train)
            
            # 4. Evaluate models
            self.logger.info("Evaluating models")
            evaluator = ModelEvaluator()
            evaluation_results = {}
            for name, model in trainer.models.items():
                evaluation_results[name] = evaluator.evaluate_model(model, X_test, y_test)
            
            # 5. Detect anomalies
            self.logger.info("Performing anomaly detection")
            anomaly_detector = AnomalyDetectorEnsemble(X_train.shape[2])
            anomaly_detector.fit(X_train)
            anomaly_results = anomaly_detector.detect_anomalies(X_test)
            
            # 6. Generate SHAP explanations
            self.logger.info("Generating model explanations")
            explainer = ExplainableAI(str(self.subdirs['shap']))
            shap_results = analyze_model_explanability(
                trainer.models,
                X_train,
                X_test,
                self.config['feature_names']
            )
            
            # 7. Compile and save results
            results = {
                'training_history': training_results,
                'evaluation_results': evaluation_results,
                'anomaly_results': anomaly_results,
                'shap_results': shap_results
            }
            
            self.save_results(results, 'analysis_results.json')
            
            # 8. Generate plots
            self.plot_model_performance(results)
            
            # 9. Generate recommendations
            recommendations = self.generate_recommendations(results)
            self.save_results(recommendations, 'recommendations.json')
            
            self.logger.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
            
    def generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate actionable recommendations based on analysis results.
        
        Args:
            results: Dictionary containing analysis results
        Returns:
            Dictionary containing recommendations
        """
        recommendations = {
            'model_selection': {},
            'anomaly_detection': {},
            'system_optimization': {}
        }
        
        # Model selection recommendations
        best_model = max(
            results['evaluation_results'].items(),
            key=lambda x: x[1]['r2']
        )[0]
        
        recommendations['model_selection'] = {
            'best_model': best_model,
            'performance_metrics': results['evaluation_results'][best_model],
            'reason': f"Selected based on highest R² score of {results['evaluation_results'][best_model]['r2']:.3f}"
        }
        
        # Anomaly detection recommendations
        anomaly_count = sum(results['anomaly_results']['ensemble_predictions'])
        recommendations['anomaly_detection'] = {
            'anomaly_count': int(anomaly_count),
            'anomaly_percentage': float(anomaly_count / len(results['anomaly_results']['ensemble_predictions']) * 100),
            'suggestion': "Investigate system behavior during identified anomaly periods"
        }
        
        # System optimization recommendations
        if 'optimization_results' in results:
            opt = results['optimization_results']
            recommendations['system_optimization'] = {
                'optimal_temperature': float(opt['optimal_temp']),
                'estimated_savings': float(opt['min_power']),
                'implementation_steps': [
                    "Gradually adjust setpoints to optimal values",
                    "Monitor system performance during transition",
                    "Validate energy savings after implementation"
                ]
            }
            
        return recommendations

def preprocess_data(X, y, sequence_length=24, forecast_horizon=12):
    # ...existing code...
    num_samples = y.shape[0] // (sequence_length * forecast_horizon)
    y = y[:num_samples * sequence_length * forecast_horizon]  # Trim y to be divisible by sequence_length * forecast_horizon
    y = y.reshape(num_samples, sequence_length, forecast_horizon)  # Ensure the target data has the correct shape
    # ...existing code...
    return X, y

if __name__ == "__main__":
    # Configuration
    config = {
        'data_path': 'D:\PowerAmp\HVAC_dataset.csv',
        'output_base_dir': 'outputs',
        'feature_names': [
            'on_off', 'damper', 'co2_1', 'amb_humid_1', 'pot_gen',
            'high_pressure_1', 'high_pressure_2', 'low_pressure_1',
            'low_pressure_2', 'high_pressure_3', 'low_pressure_3',
            'outside_temp', 'outlet_temp', 'inlet_temp',
            'summer_setpoint_temp', 'winter_setpoint_temp', 'amb_temp_2'
        ],
        'model_params': {
            'sequence_length': 24,
            'forecast_horizon': 12,
            'batch_size': 32,
            'epochs': 50
        }
    }
    
    # Run pipeline
    pipeline = HVACAnalysisPipeline(config)
    results = pipeline.run_pipeline()