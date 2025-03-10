# Analysis Report
## Model Performance
- Mean Absolute Error (MAE): 0.0410
- Root Mean Squared Error (RMSE): 0.0545
- R² Score: 0.9969

## Optimal Settings
No optimal settings found within the specified ranges.

## Anomalies Detected
- Total anomalies detected by Autoencoder: 500
- Total anomalies detected by Isolation Forest: 0

## Cross-Validation Results
### lstm
- MAE: 0.0490 ± 0.0100
- RMSE: 0.0632 ± 0.0119
- R²: 0.9931 ± 0.0017
### advanced_lstm
- MAE: nan ± nan
- RMSE: nan ± nan
- R²: nan ± nan
### bi_lstm
- MAE: 0.0935 ± 0.0389
- RMSE: 0.1155 ± 0.0394
- R²: 0.9771 ± 0.0103
### attention
- MAE: 0.0891 ± 0.0330
- RMSE: 0.1078 ± 0.0360
- R²: 0.9798 ± 0.0090
### combined
- MAE: 0.1188 ± 0.0154
- RMSE: 0.1432 ± 0.0175
- R²: 0.9638 ± 0.0100

## Feature Importance
                feature  importance
 Power Factor * Current    0.954309
            Current (A)    0.024082
        Current Squared    0.023826
    Humidity Difference    0.014200
Inside Temperature (°C)    0.011432
 Temperature Difference    0.010494