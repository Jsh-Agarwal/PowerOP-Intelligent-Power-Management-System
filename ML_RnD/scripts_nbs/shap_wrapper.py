import shap
import numpy as np
import logging
from typing import Any, Optional

class ShapWrapper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def create_explainer(self, model: Any, X_background: np.ndarray) -> Any:
        """Creates SHAP explainer based on available SHAP version"""
        try:
            # Try newer SHAP API first
            return shap.Explainer(model, X_background)
        except AttributeError:
            try:
                # Try older SHAP API
                return shap.KernelExplainer(model.predict, X_background)
            except Exception as e:
                self.logger.error(f"Failed to create SHAP explainer: {e}")
                raise
                
    def explain_model(self, 
                     model: Any,
                     X_train: np.ndarray,
                     X_test: np.ndarray,
                     feature_names: Optional[list] = None,
                     n_samples: int = 100) -> dict:
        """Compute SHAP values and return explanation results"""
        try:
            # Sample background data
            if len(X_train) > n_samples:
                idx = np.random.choice(len(X_train), n_samples, replace=False)
                X_background = X_train[idx]
            else:
                X_background = X_train
                
            # Create explainer
            explainer = self.create_explainer(model, X_background)
            
            # Get SHAP values
            shap_values = explainer.shap_values(X_test)
            
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
                
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(0)
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
                
            return {
                'shap_values': shap_values,
                'feature_importance': dict(zip(feature_names, feature_importance))
            }
            
        except Exception as e:
            self.logger.error(f"Error in SHAP explanation: {e}")
            return {
                'shap_values': None,
                'feature_importance': None,
                'error': str(e)
            }
