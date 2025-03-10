import numpy as np
import pandas as pd
import logging
import json
import os
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from regression import ModelTrainer, ModelEvaluator
from anomaly import AnomalyDetectorEnsemble
from shap import analyze_model_explanability
from preprocess_data import preprocess_data

class PowerConsumptionAnalyzer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            filename=f'{output_dir}/analysis.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def run_analysis(self, data_paths: List[str]) -> Dict[str, Any]:
        """Runs the complete analysis pipeline."""
        try:
            logging.info(f"Starting analysis for {data_paths}")
            data, features, target = preprocess_data(data_paths)
            X = data[features]
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            trainer = ModelTrainer(input_shape=(1, len(features)))
            trainer.compile_models()
            trainer.train_all_models(X_train, y_train)

            for name, model in trainer.models.items():
                model.save(f'{self.output_dir}/{name}_model.keras')

            evaluator = ModelEvaluator()
            evaluation_results = {
                name: evaluator.evaluate_model(model, X_test, y_test)
                for name, model in trainer.models.items()
            }

            anomaly_detector = AnomalyDetectorEnsemble(input_dim=len(features))
            anomaly_detector.fit(X_train)
            anomaly_predictions = anomaly_detector.detect_anomalies(X_test)

            shap_results = analyze_model_explanability(
                trainer.models, X_train.values, X_test.values, features, self.output_dir
            )

            best_model_name = max(evaluation_results, key=lambda k: evaluation_results[k]['r2'])
            best_model = trainer.models[best_model_name]
            optimization_results = evaluator.find_optimal_settings(
                best_model, anomaly_detector.detectors['complex_autoencoder'].model, features, features
            )

            self.save_results({
                'model_evaluation': evaluation_results,
                'anomaly_detection': anomaly_predictions.tolist(),
                'shap_analysis': shap_results,
                'optimization': optimization_results
            })

            logging.info(f"Analysis completed for {data_paths}")
            return {
                'models': trainer.models,
                'evaluation': evaluation_results,
                'anomalies': anomaly_predictions,
                'explanations': shap_results,
                'optimal_settings': optimization_results
            }
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            raise

    def save_results(self, results: Dict[str, Any]):
        """Saves analysis results as JSON and CSV files."""
        try:
            with open(f'{self.output_dir}/evaluation_results.json', 'w') as f:
                json.dump(results['model_evaluation'], f, indent=4)

            with open(f'{self.output_dir}/optimal_settings.json', 'w') as f:
                json.dump(results['optimization'], f, indent=4)

            pd.DataFrame(results['shap_analysis']['feature_importance']).to_csv(
                f'{self.output_dir}/feature_importance.csv', index=False
            )
            logging.info("Results saved successfully")
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")
            raise

if __name__ == "__main__":
    analyzer = PowerConsumptionAnalyzer("results")
    datasets = [
        "data/Building_Base_Cooling.csv", "data/Building_Base_Heating.csv",
        "data/Building_FF_Cooling.csv", "data/Building_FF_Heating.csv",
        "data/Building_Pre_Heating.csv", "data/Building_SB_Cooling.csv",
        "data/Building_SB_Heating.csv", "data/Weather_Base_Cooling.csv",
        "data/Weather_Base_Heating.csv", "data/Weather_FF_Cooling.csv",
        "data/Weather_FF_Heating.csv", "data/Weather_Pre_Heating.csv",
        "data/Weather_SB_Cooling.csv", "data/Weather_SB_Heating.csv"
    ]
    results = analyzer.run_analysis(datasets)
