import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from scipy.optimize import minimize

from ..services.weather_service import WeatherService
from ..services.astra_db_service import AstraDBService
from ..models.lstm_model import LSTMModel
from .comfort_optimization import ComfortOptimizer
from ..utils.exceptions import OptimizationError

@dataclass
class EnergyMetrics:
    """Energy consumption and cost metrics container."""
    total_consumption: float  # kWh
    peak_demand: float  # kW
    base_load: float  # kW
    efficiency_ratio: float  # Actual vs. Expected efficiency
    cost_per_kwh: float  # Average cost per kWh
    demand_charges: float  # Peak demand charges
    total_cost: float  # Total energy cost
    carbon_footprint: float  # CO2 emissions in kg

class TimeOfUseRates:
    """Time-of-use electricity rate structure."""
    def __init__(
        self,
        peak_rate: float,
        mid_peak_rate: float,
        off_peak_rate: float,
        demand_charge: float
    ):
        self.peak_rate = peak_rate
        self.mid_peak_rate = mid_peak_rate
        self.off_peak_rate = off_peak_rate
        self.demand_charge = demand_charge  # per kW

class EnergyOptimizer:
    """HVAC system energy optimization engine."""
    
    def __init__(
        self,
        db_service: Optional[AstraDBService] = None,
        weather_service: Optional[WeatherService] = None,
        lstm_model: Optional[LSTMModel] = None,
        comfort_optimizer: Optional[ComfortOptimizer] = None
    ):
        """Initialize energy optimizer with required services."""
        self.db = db_service or AstraDBService()
        self.weather = weather_service or WeatherService()
        self.lstm_model = lstm_model
        self.comfort_optimizer = comfort_optimizer or ComfortOptimizer()
        
        # Default optimization parameters
        self.min_temperature = 18.0  # °C
        self.max_temperature = 26.0  # °C
        self.min_comfort_score = 70  # Minimum acceptable comfort score
        
        # Time-of-use rate structure (example rates)
        self.rate_structure = TimeOfUseRates(
            peak_rate=0.25,      # $/kWh during peak hours
            mid_peak_rate=0.15,  # $/kWh during mid-peak hours
            off_peak_rate=0.08,  # $/kWh during off-peak hours
            demand_charge=15.0   # $/kW of peak demand
        )

    async def analyze_energy_consumption(
        self,
        system_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> EnergyMetrics:
        """Analyze historical energy consumption patterns."""
        try:
            # Get historical consumption data
            consumption_data = await self.db.get_system_status(
                system_id=system_id,
                start_time=start_time,
                end_time=end_time
            )
            
            if not consumption_data:
                raise OptimizationError("No historical data available")
            
            # Calculate energy metrics
            df = pd.DataFrame([s.to_dict() for s in consumption_data])
            
            total_consumption = df['energy_consumption'].sum()
            peak_demand = df['active_power'].max()
            base_load = df['active_power'].quantile(0.1)  # 10th percentile as base load
            
            # Calculate efficiency ratio
            expected_consumption = self._calculate_expected_consumption(df)
            efficiency_ratio = total_consumption / expected_consumption if expected_consumption > 0 else 0
            
            # Calculate costs
            costs = self._calculate_energy_costs(df, self.rate_structure)
            
            return EnergyMetrics(
                total_consumption=total_consumption,
                peak_demand=peak_demand,
                base_load=base_load,
                efficiency_ratio=efficiency_ratio,
                cost_per_kwh=costs['total_cost'] / total_consumption if total_consumption > 0 else 0,
                demand_charges=costs['demand_charges'],
                total_cost=costs['total_cost'],
                carbon_footprint=self._calculate_carbon_footprint(total_consumption)
            )
            
        except Exception as e:
            raise OptimizationError(f"Energy consumption analysis failed: {str(e)}")

    async def optimize_energy_cost(
        self,
        system_id: str,
        forecast_hours: int = 24,
        comfort_constraint: bool = True
    ) -> Dict[str, Any]:
        """Generate cost-optimized operation schedule."""
        try:
            # Get weather forecast
            weather_forecast = await self.weather.get_forecast(system_id, days=1)
            
            # Predict future load using LSTM model
            load_prediction = await self._predict_future_load(
                system_id,
                weather_forecast,
                forecast_hours
            )
            
            # Define optimization constraints
            constraints = self._define_optimization_constraints(
                comfort_constraint=comfort_constraint
            )
            
            # Optimize schedule using predicted load
            optimal_schedule = self._optimize_operation_schedule(
                load_prediction,
                weather_forecast,
                constraints
            )
            
            # Calculate expected savings
            savings = self._calculate_projected_savings(
                current_schedule=load_prediction,
                optimal_schedule=optimal_schedule
            )
            
            return {
                "optimal_schedule": optimal_schedule,
                "projected_savings": savings,
                "recommendations": self._generate_recommendations(
                    load_prediction,
                    optimal_schedule,
                    weather_forecast
                )
            }
            
        except Exception as e:
            raise OptimizationError(f"Energy cost optimization failed: {str(e)}")

    async def manage_peak_load(
        self,
        system_id: str,
        current_load: float,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Manage peak load to reduce demand charges."""
        try:
            # Get historical peak data
            historical_peaks = await self._get_historical_peaks(system_id)
            
            # Calculate optimal threshold if not provided
            if threshold is None:
                threshold = self._calculate_optimal_peak_threshold(historical_peaks)
            
            # Check if current load approaches threshold
            if current_load > threshold * 0.9:  # 90% of threshold
                # Generate load reduction strategies
                reduction_strategies = self._generate_load_reduction_strategies(
                    current_load,
                    threshold,
                    historical_peaks
                )
                
                return {
                    "alert_level": "high",
                    "current_load": current_load,
                    "threshold": threshold,
                    "recommendations": reduction_strategies
                }
            
            return {
                "alert_level": "normal",
                "current_load": current_load,
                "threshold": threshold,
                "recommendations": []
            }
            
        except Exception as e:
            raise OptimizationError(f"Peak load management failed: {str(e)}")

    def _calculate_expected_consumption(self, df: pd.DataFrame) -> float:
        """Calculate expected energy consumption based on conditions."""
        # Implement regression model or engineering calculations
        # This is a simplified example
        heating_degree_hours = np.maximum(20 - df['outside_temp'], 0).sum()
        cooling_degree_hours = np.maximum(df['outside_temp'] - 24, 0).sum()
        
        # Coefficients determined from historical data analysis
        heating_coefficient = 0.5  # kWh per degree-hour
        cooling_coefficient = 0.7  # kWh per degree-hour
        base_load = 100  # Base load in kWh
        
        return (heating_degree_hours * heating_coefficient +
                cooling_degree_hours * cooling_coefficient + base_load)

    def _calculate_energy_costs(
        self,
        df: pd.DataFrame,
        rates: TimeOfUseRates
    ) -> Dict[str, float]:
        """Calculate energy costs including time-of-use rates."""
        # Add hour column if not present
        if 'hour' not in df.columns:
            df['hour'] = pd.to_datetime(df.index).hour
        
        # Define peak periods
        peak_hours = (df['hour'].between(14, 19))  # 2 PM - 7 PM
        mid_peak_hours = (df['hour'].between(8, 13) | 
                         df['hour'].between(20, 22))  # 8 AM - 1 PM & 8 PM - 10 PM
        off_peak_hours = ~(peak_hours | mid_peak_hours)
        
        # Calculate energy costs
        peak_cost = df.loc[peak_hours, 'energy_consumption'].sum() * rates.peak_rate
        mid_peak_cost = df.loc[mid_peak_hours, 'energy_consumption'].sum() * rates.mid_peak_rate
        off_peak_cost = df.loc[off_peak_hours, 'energy_consumption'].sum() * rates.off_peak_rate
        
        # Calculate demand charges
        demand_charges = df['active_power'].max() * rates.demand_charge
        
        return {
            'peak_cost': peak_cost,
            'mid_peak_cost': mid_peak_cost,
            'off_peak_cost': off_peak_cost,
            'demand_charges': demand_charges,
            'total_cost': peak_cost + mid_peak_cost + off_peak_cost + demand_charges
        }

    def _calculate_carbon_footprint(self, total_consumption: float) -> float:
        """Calculate CO2 emissions from energy consumption."""
        # Average CO2 emissions factor (kg CO2 per kWh)
        # This varies by region and energy source mix
        emissions_factor = 0.4  # Example value
        return total_consumption * emissions_factor

    async def _predict_future_load(
        self,
        system_id: str,
        weather_forecast: Dict[str, Any],
        forecast_hours: int
    ) -> np.ndarray:
        """Predict future energy load using LSTM model."""
        if not self.lstm_model:
            # Initialize LSTM model if not provided
            self.lstm_model = LSTMModel(input_shape=(24, 5))  # Example shape
            self.lstm_model.load_model('models/energy_lstm.h5')
        
        # Prepare input features
        features = self._prepare_prediction_features(
            system_id,
            weather_forecast,
            forecast_hours
        )
        
        # Make prediction
        predictions = await self.lstm_model.predict_next_24h(
            features,
            weather_forecast
        )
        
        return predictions['predictions']

    def _optimize_operation_schedule(
        self,
        load_prediction: np.ndarray,
        weather_forecast: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize HVAC operation schedule for minimum cost."""
        def objective(x):
            """Objective function for optimization."""
            return self._calculate_total_cost(x, self.rate_structure)
        
        # Initial schedule is the predicted load
        x0 = load_prediction
        
        # Define bounds for optimization variables
        bounds = [(constraints['min_load'], constraints['max_load'])] * len(x0)
        
        # Optimize schedule
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=self._get_optimization_constraints(constraints)
        )
        
        if not result.success:
            raise OptimizationError("Failed to find optimal schedule")
            
        return {
            "schedule": result.x,
            "expected_cost": result.fun,
            "convergence": result.success
        }

    def _generate_recommendations(
        self,
        current_schedule: np.ndarray,
        optimal_schedule: Dict[str, Any],
        weather_forecast: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from optimization results."""
        recommendations = []
        
        # Analyze peak shifting opportunities
        peak_shifts = self._identify_peak_shifts(
            current_schedule,
            optimal_schedule['schedule']
        )
        
        for shift in peak_shifts:
            recommendations.append({
                "type": "peak_shift",
                "from_hour": shift['from_hour'],
                "to_hour": shift['to_hour'],
                "load_reduction": shift['reduction'],
                "cost_saving": shift['saving']
            })
        
        # Analyze efficiency improvements
        efficiency_gaps = self._identify_efficiency_gaps(
            current_schedule,
            weather_forecast
        )
        
        for gap in efficiency_gaps:
            recommendations.append({
                "type": "efficiency_improvement",
                "system": gap['system'],
                "potential_saving": gap['saving'],
                "payback_period": gap['payback'],
                "priority": gap['priority']
            })
        
        return recommendations

    def _identify_peak_shifts(
        self,
        current: np.ndarray,
        optimal: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Identify opportunities for peak load shifting."""
        shifts = []
        peak_threshold = np.percentile(current, 75)  # 75th percentile
        
        for hour in range(len(current)):
            if current[hour] > peak_threshold and optimal[hour] < current[hour]:
                # Find nearest off-peak hour
                shift_to = self._find_best_shift_hour(hour, optimal)
                
                if shift_to is not None:
                    shifts.append({
                        "from_hour": hour,
                        "to_hour": shift_to,
                        "reduction": float(current[hour] - optimal[hour]),
                        "saving": self._calculate_shift_saving(
                            current[hour] - optimal[hour],
                            hour,
                            shift_to
                        )
                    })
        
        return shifts

    def _identify_efficiency_gaps(
        self,
        current_load: np.ndarray,
        weather_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify system efficiency improvement opportunities."""
        gaps = []
        
        # Calculate baseline efficiency
        baseline_efficiency = self._calculate_system_efficiency(
            current_load,
            weather_data
        )
        
        # Check different subsystems
        systems = {
            "compressor": {"expected": 0.85, "improvement_cost": 5000},
            "fans": {"expected": 0.80, "improvement_cost": 3000},
            "heat_exchanger": {"expected": 0.90, "improvement_cost": 4000}
        }
        
        for system, params in systems.items():
            current_eff = baseline_efficiency.get(system, 0)
            if current_eff < params["expected"]:
                potential_saving = self._calculate_efficiency_saving(
                    current_eff,
                    params["expected"],
                    current_load.mean()
                )
                
                payback = params["improvement_cost"] / potential_saving if potential_saving > 0 else float('inf')
                
                gaps.append({
                    "system": system,
                    "current_efficiency": current_eff,
                    "target_efficiency": params["expected"],
                    "saving": potential_saving,
                    "payback": payback,
                    "priority": "high" if payback < 2 else "medium" if payback < 4 else "low"
                })
        
        return gaps

    async def close(self):
        """Close service connections."""
        try:
            self.db.close()
            await self.weather.close()
            if self.comfort_optimizer:
                self.comfort_optimizer.close()
        except Exception as e:
            logger.error(f"Error closing services: {str(e)}")
