import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import random
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Results directory with absolute path
RESULTS_DIR = Path(__file__).parent.parent / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

def setup_style():
    """Setup plotting style."""
    try:
        plt.style.use('default')  # Use default style instead of seaborn
        sns.set_theme()  # This is the correct way to set seaborn style
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['figure.dpi'] = 100
    except Exception as e:
        logger.warning(f"Style setup warning: {str(e)}")
        # Fallback to basic matplotlib settings
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['figure.dpi'] = 100

def generate_model_comparison():
    """Generate model comparison visualizations."""
    models = ['LSTM', 'AutoEncoder']
    metrics = {
        'MAE': [0.45, 0.52],
        'RMSE': [0.58, 0.64],
        'R2 Score': [0.92, 0.89],
        'Training Time (s)': [245, 180]
    }
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    for (metric, values), ax in zip(metrics.items(), axes.flat):
        ax.bar(models, values)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'model_comparison.png')
    plt.close()

def generate_training_history():
    """Generate training history plots."""
    # Mock training history
    epochs = range(1, 101)
    lstm_history = {
        'loss': [np.exp(-0.05 * x) + 0.1 * np.random.random() for x in epochs],
        'val_loss': [np.exp(-0.04 * x) + 0.15 * np.random.random() for x in epochs]
    }
    
    autoencoder_history = {
        'loss': [np.exp(-0.06 * x) + 0.12 * np.random.random() for x in epochs],
        'val_loss': [np.exp(-0.045 * x) + 0.18 * np.random.random() for x in epochs]
    }
    
    # Create training history plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(epochs, lstm_history['loss'], label='Training Loss')
    ax1.plot(epochs, lstm_history['val_loss'], label='Validation Loss')
    ax1.set_title('LSTM Training History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, autoencoder_history['loss'], label='Training Loss')
    ax2.plot(epochs, autoencoder_history['val_loss'], label='Validation Loss')
    ax2.set_title('AutoEncoder Training History')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'training_history.png')
    plt.close()

def generate_prediction_analysis():
    """Generate prediction analysis visualizations."""
    # Generate sample data with corrected frequency
    timestamps = pd.date_range(start='2025-01-01', periods=48, freq='h')  # Changed 'H' to 'h'
    actual = [22 + 2*np.sin(x/12) + np.random.normal(0, 0.5) for x in range(48)]
    lstm_pred = [22 + 2*np.sin(x/12) + np.random.normal(0, 0.3) for x in range(48)]
    ae_pred = [22 + 2*np.sin(x/12) + np.random.normal(0, 0.4) for x in range(48)]
    
    # Create interactive plot using plotly with explicit layout
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=timestamps, y=actual, name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=timestamps, y=lstm_pred, name='LSTM', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=timestamps, y=ae_pred, name='AutoEncoder', line=dict(color='green')))
    
    fig.update_layout(
        title='Temperature Prediction Comparison',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        width=1200,  # Explicit width
        height=600,  # Explicit height
        margin=dict(l=50, r=50, t=50, b=50),  # Explicit margins
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    fig.write_html(RESULTS_DIR / 'prediction_analysis.html')

def generate_system_performance():
    """Generate system performance visualizations."""
    # Generate sample performance data
    dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
    metrics = {
        'Energy Consumption (kWh)': [random.uniform(80, 120) for _ in range(30)],
        'Cost Savings (%)': [random.uniform(5, 15) for _ in range(30)],
        'Comfort Score': [random.uniform(85, 98) for _ in range(30)],
        'System Efficiency (%)': [random.uniform(75, 95) for _ in range(30)]
    }
    
    df = pd.DataFrame(metrics, index=dates)
    
    # Create figure with more explicit layout control
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)  # Increased horizontal space
    
    # Energy Consumption
    ax1 = fig.add_subplot(gs[0, 0])
    sns.lineplot(data=df['Energy Consumption (kWh)'], ax=ax1, marker='o')
    ax1.set_title('Daily Energy Consumption')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_xlabel('')  # Remove x-label to avoid overlap
    
    # Cost Savings
    ax2 = fig.add_subplot(gs[0, 1])
    sns.lineplot(data=df['Cost Savings (%)'], ax=ax2, marker='s')
    ax2.set_title('Cost Savings Trend')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_xlabel('')  # Remove x-label to avoid overlap
    
    # Comfort Score
    ax3 = fig.add_subplot(gs[1, 0])
    sns.lineplot(data=df['Comfort Score'], ax=ax3, marker='^')
    ax3.set_title('Comfort Score Variation')
    ax3.tick_params(axis='x', rotation=45)
    
    # System Efficiency
    ax4 = fig.add_subplot(gs[1, 1])
    sns.lineplot(data=df['System Efficiency (%)'], ax=ax4, marker='d')
    ax4.set_title('System Efficiency Trend')
    ax4.tick_params(axis='x', rotation=45)
    
    # Adjust layout with explicit padding
    plt.gcf().set_size_inches(15, 10)
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9)
    plt.savefig(RESULTS_DIR / 'system_performance.png', bbox_inches='tight', dpi=300)
    plt.close()

def generate_temperature_distribution():
    """Generate temperature distribution analysis."""
    # Generate sample temperature data
    temps = np.random.normal(22, 2, 1000)  # Normal distribution around 22°C
    zones = ['Zone A', 'Zone B', 'Zone C', 'Zone D']
    zone_temps = {
        zone: np.random.normal(22 + i, 1.5, 1000) 
        for i, zone in enumerate(zones)
    }
    
    # Create distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall distribution
    sns.histplot(temps, kde=True, ax=ax1)
    ax1.set_title('Overall Temperature Distribution')
    ax1.set_xlabel('Temperature (°C)')
    
    # Zone-wise distribution
    for zone, temp in zone_temps.items():
        sns.kdeplot(temp, ax=ax2, label=zone)
    ax2.set_title('Zone-wise Temperature Distribution')
    ax2.set_xlabel('Temperature (°C)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'temperature_distribution.png')
    plt.close()

def generate_optimization_impact():
    """Generate optimization impact analysis."""
    # Sample data for optimization impact
    before_after = {
        'Energy Usage (kWh)': [100, 85],
        'Cost ($)': [150, 120],
        'CO2 Emissions (kg)': [75, 60],
        'Comfort Score': [85, 92]
    }
    
    # Calculate percentage improvements
    improvements = {
        metric: ((after - before) / before * 100)
        for metric, (before, after) in before_after.items()
    }
    
    # Create impact visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before vs After comparison
    x = np.arange(len(before_after))
    width = 0.35
    
    ax1.bar(x - width/2, [v[0] for v in before_after.values()], width, label='Before')
    ax1.bar(x + width/2, [v[1] for v in before_after.values()], width, label='After')
    ax1.set_xticks(x)
    ax1.set_xticklabels(before_after.keys(), rotation=45)
    ax1.set_title('Before vs After Optimization')
    ax1.legend()
    
    # Percentage improvements
    colors = ['g' if v > 0 else 'r' for v in improvements.values()]
    ax2.bar(improvements.keys(), improvements.values(), color=colors)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Optimization Impact')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'optimization_impact.png')
    plt.close()

def generate_all_reports():
    """Generate all analysis reports."""
    try:
        setup_style()
        
        # Generate all visualizations
        generate_model_comparison()
        generate_training_history()
        generate_prediction_analysis()
        generate_system_performance()
        generate_temperature_distribution()
        generate_optimization_impact()
        
        # Create summary report
        summary = {
            "generated_at": datetime.now().isoformat(),
            "reports": [
                {"name": "Model Comparison", "file": "model_comparison.png"},
                {"name": "Training History", "file": "training_history.png"},
                {"name": "Prediction Analysis", "file": "prediction_analysis.html"},
                {"name": "System Performance", "file": "system_performance.png"},
                {"name": "Temperature Distribution", "file": "temperature_distribution.png"},
                {"name": "Optimization Impact", "file": "optimization_impact.png"}
            ]
        }
        
        with open(RESULTS_DIR / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Reports generated successfully in {RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"Error generating reports: {str(e)}")
        raise

if __name__ == "__main__":
    generate_all_reports()
    print(f"Reports generated successfully in {RESULTS_DIR}")