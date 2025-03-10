import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import logging
from typing import Dict, Union, Tuple, List
import os
from sklearn.base import BaseEstimator


def analyze_model_explanability(
    models: Dict[str, BaseEstimator],
    X_train: np.ndarray,
    X_test: np.ndarray,
    features: List[str],
    output_dir: str
) -> Dict:
    """
    Comprehensive analysis of model explanability for both supervised and unsupervised models.
    
    Returns:
    - Dictionary containing feature importance and SHAP values for each model
    """
    explainer = ExplainableAI(output_dir)
    results = {}
    
    for model_name, model in models.items():
        try:
            # Create explainer and compute SHAP values
            explainer.create_explainer(model, X_train, model_name)
            shap_values = explainer.compute_shap_values(X_test, model_name)
            
            # Generate importance analysis
            importance_df = explainer.generate_feature_importance(model_name, features)
            
            # Create visualizations
            explainer.plot_shap_summary(model_name, features, X_test)
            explainer.plot_feature_dependence(model_name, features, X_test)
            
            results[model_name] = {
                'shap_values': shap_values,
                'feature_importance': importance_df
            }
            
        except Exception as e:
            logging.error(f"Error analyzing model {model_name}: {str(e)}")
    
    return results

def build_complex_autoencoder(input_dim):
    """Enhanced autoencoder with additional layers and normalization"""
    input_layer = Input(shape=(input_dim,))
    # Encoder
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = LayerNormalization()(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(32, activation='relu')(encoded)
    decoded = LayerNormalization()(decoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return autoencoder
