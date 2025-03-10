import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass

from ..services.weather_service import WeatherService
from ..services.astra_db_service import AstraDBService
from ..services.groq_slm_service import GroqSLMService
from ..utils.exceptions import OptimizationError

@dataclass
class ComfortMetrics:
    """Comfort metrics container."""
    pmv: float  # Predicted Mean Vote (-3 to +3)
    ppd: float  # Predicted Percentage Dissatisfied (%)
    co2_level: float  # CO2 level (ppm)
    air_quality: float  # Air quality index (0-100)
    temperature_deviation: float  # Deviation from preferred temperature
    humidity_deviation: float  # Deviation from preferred humidity

class ComfortOptimizer:
    """HVAC system comfort optimization engine."""
    
    def __init__(
        self,
        db_service: Optional[AstraDBService] = None,
        weather_service: Optional[WeatherService] = None,
        groq_service: Optional[GroqSLMService] = None
    ):
        """Initialize comfort optimizer with required services."""
        self.db = db_service or AstraDBService()
        self.weather = weather_service or WeatherService()
        self.groq = groq_service or GroqSLMService()
        
        # Comfort parameters
        self.temp_weight = 0.4
        self.humidity_weight = 0.3
        self.air_quality_weight = 0.3
        self.metabolic_rate = 1.2  # Standard office activity
        self.clothing_insulation = 1.0  # Standard indoor clothing
        
        # Optimal ranges
        self.optimal_temp_range = (20.0, 24.0)  # °C
        self.optimal_humidity_range = (40.0, 60.0)  # %
        self.optimal_co2_range = (400, 1000)  # ppm
        
    async def calculate_comfort_metrics(
        self,
        temperature: float,
        humidity: float,
        co2_level: float,
        air_speed: float,
        mean_radiant_temp: Optional[float] = None
    ) -> ComfortMetrics:
        """
        Calculate comprehensive comfort metrics using Fanger's PMV model.
        
        Args:
            temperature: Air temperature in °C
            humidity: Relative humidity in %
            co2_level: CO2 concentration in ppm
            air_speed: Air speed in m/s
            mean_radiant_temp: Mean radiant temperature in °C
            
        Returns:
            ComfortMetrics object with calculated metrics
        """
        try:
            # Use mean radiant temp if provided, otherwise use air temperature
            mrt = mean_radiant_temp if mean_radiant_temp is not None else temperature
            
            # Calculate PMV (Predicted Mean Vote)
            pmv = self._calculate_pmv(
                temperature,
                mrt,
                humidity,
                air_speed,
                self.metabolic_rate,
                self.clothing_insulation
            )
            
            # Calculate PPD (Predicted Percentage Dissatisfied)
            ppd = 100 - 95 * np.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
            
            # Calculate air quality index (0-100)
            air_quality = self._calculate_air_quality_index(co2_level)
            
            # Calculate deviations from optimal ranges
            temp_deviation = min(
                abs(temperature - self.optimal_temp_range[0]),
                abs(temperature - self.optimal_temp_range[1])
            )
            
            humidity_deviation = min(
                abs(humidity - self.optimal_humidity_range[0]),
                abs(humidity - self.optimal_humidity_range[1])
            )
            
            return ComfortMetrics(
                pmv=pmv,
                ppd=ppd,
                co2_level=co2_level,
                air_quality=air_quality,
                temperature_deviation=temp_deviation,
                humidity_deviation=humidity_deviation
            )
            
        except Exception as e:
            raise OptimizationError(f"Failed to calculate comfort metrics: {str(e)}")

    def _calculate_pmv(
        self,
        ta: float,  # Air temperature
        tr: float,  # Mean radiant temperature
        rh: float,  # Relative humidity
        va: float,  # Air velocity
        met: float,  # Metabolic rate
        clo: float  # Clothing insulation
    ) -> float:
        """
        Calculate Predicted Mean Vote using Fanger's thermal comfort model.
        """
        # Constants
        M = met * 58.15  # Metabolic rate in W/m²
        W = 0  # External work, normally 0
        ICL = 0.155 * clo  # Thermal insulation of clothing in m²K/W
        
        # Calculate water vapor pressure
        pa = rh * 10 * np.exp(16.6536 - 4030.183 / (ta + 235))
        
        # Calculate surface temperature of clothing
        fcl = 1.05 + 0.645 * ICL
        hc = 12.1 * np.sqrt(va)
        
        tcl = ta + (35.5 - ta) / (3.5 * (6.45 * ICL + 0.1))
        
        # Radiative heat transfer coefficient
        hr = 4 * 0.9 * 5.67e-8 * ((tcl + 273)**4 - (tr + 273)**4) / (tcl - tr)
        
        # Calculate PMV
        L = M - W - 3.96e-8 * fcl * ((tcl + 273)**4 - (tr + 273)**4) - \
            fcl * hc * (tcl - ta) - 3.05 * (5.73 - 0.007 * M - pa) - \
            0.42 * (M - 58.15) - 0.0173 * M * (5.87 - pa) - \
            0.0014 * M * (34 - ta)
            
        pmv = 0.303 * np.exp(-0.036 * M) + 0.028
        return pmv * L

    def _calculate_air_quality_index(self, co2_level: float) -> float:
        """Calculate air quality index based on CO2 levels."""
        if co2_level <= self.optimal_co2_range[0]:
            return 100
        elif co2_level >= self.optimal_co2_range[1]:
            return 0
        else:
            # Linear interpolation between optimal range
            return 100 * (1 - (co2_level - self.optimal_co2_range[0]) / 
                         (self.optimal_co2_range[1] - self.optimal_co2_range[0]))

    async def optimize_comfort(
        self,
        zone_id: str,
        user_id: str,
        current_conditions: Dict[str, float],
        energy_constraint: Optional[float] = None
    ) -> Dict[str, Union[float, str, List[Dict[str, Union[float, str]]]]]:
        """
        Generate comfort optimization recommendations.
        
        Args:
            zone_id: Zone identifier
            user_id: User identifier
            current_conditions: Current environmental conditions
            energy_constraint: Optional energy consumption constraint
            
        Returns:
            Dictionary containing optimization recommendations
        """
        try:
            # Get user preferences
            user_prefs = await self._get_user_preferences(user_id, zone_id)
            
            # Get weather forecast
            weather = await self.weather.get_current_weather(zone_id)
            
            # Calculate current comfort metrics
            metrics = await self.calculate_comfort_metrics(
                temperature=current_conditions["temperature"],
                humidity=current_conditions["humidity"],
                co2_level=current_conditions["co2_level"],
                air_speed=current_conditions["air_speed"]
            )
            
            # Generate optimization context
            context = {
                "current_conditions": current_conditions,
                "comfort_metrics": metrics.__dict__,
                "user_preferences": user_prefs,
                "weather_conditions": weather,
                "energy_constraint": energy_constraint
            }
            
            # Get Groq recommendations
            recommendations = await self.groq.generate_hvac_optimization(
                hvac_data=current_conditions,
                weather_data=weather,
                optimization_target="comfort"
            )
            
            # Process and validate recommendations
            validated_recommendations = self._validate_recommendations(
                recommendations["recommendations"],
                metrics,
                user_prefs,
                energy_constraint
            )
            
            # Store optimization results
            await self._store_optimization_results(
                zone_id=zone_id,
                user_id=user_id,
                metrics=metrics,
                recommendations=validated_recommendations
            )
            
            return {
                "comfort_score": self._calculate_comfort_score(metrics),
                "recommendations": validated_recommendations,
                "expected_improvement": self._calculate_expected_improvement(
                    metrics,
                    validated_recommendations
                ),
                "energy_impact": self._estimate_energy_impact(
                    validated_recommendations
                )
            }
            
        except Exception as e:
            raise OptimizationError(f"Comfort optimization failed: {str(e)}")

    async def _get_user_preferences(
        self,
        user_id: str,
        zone_id: str
    ) -> Dict[str, Any]:
        """Get user comfort preferences from database."""
        try:
            preferences = await self.db.get_user_preferences(
                user_id=user_id,
                zone_id=zone_id
            )
            
            if not preferences:
                return self._get_default_preferences()
                
            return preferences[0].to_dict()
            
        except Exception as e:
            raise OptimizationError(f"Failed to get user preferences: {str(e)}")

    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default comfort preferences."""
        return {
            "preferred_temperature": 22.0,
            "preferred_humidity": 50.0,
            "temperature_tolerance": 2.0,
            "humidity_tolerance": 10.0,
            "air_quality_sensitivity": "medium",
            "energy_saving_priority": "medium"
        }

    def _validate_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        metrics: ComfortMetrics,
        user_prefs: Dict[str, Any],
        energy_constraint: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Validate and filter optimization recommendations."""
        validated = []
        
        for rec in recommendations:
            # Skip if recommendation violates energy constraint
            if energy_constraint and rec.get("energy_impact", 0) > energy_constraint:
                continue
                
            # Skip if recommendation moves away from user preferences
            if not self._check_preference_compliance(rec, user_prefs):
                continue
                
            # Add validation status
            rec["validated"] = True
            validated.append(rec)
            
        return validated

    def _calculate_comfort_score(self, metrics: ComfortMetrics) -> float:
        """Calculate overall comfort score (0-100)."""
        # Convert PMV to comfort score (0-100)
        pmv_score = 100 * (1 - abs(metrics.pmv) / 3)
        
        # Convert PPD to comfort score (0-100)
        ppd_score = 100 - metrics.ppd
        
        # Air quality score already in 0-100 range
        air_score = metrics.air_quality
        
        # Weighted average of scores
        return (
            self.temp_weight * pmv_score +
            self.humidity_weight * ppd_score +
            self.air_quality_weight * air_score
        )

    def _calculate_expected_improvement(
        self,
        current_metrics: ComfortMetrics,
        recommendations: List[Dict[str, Any]]
    ) -> float:
        """Calculate expected comfort improvement from recommendations."""
        current_score = self._calculate_comfort_score(current_metrics)
        
        # Simulate metrics after applying recommendations
        simulated_metrics = self._simulate_recommendations_impact(
            current_metrics,
            recommendations
        )
        
        improved_score = self._calculate_comfort_score(simulated_metrics)
        return improved_score - current_score

    def _estimate_energy_impact(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> float:
        """Estimate energy impact of recommendations."""
        total_impact = 0.0
        
        for rec in recommendations:
            # Get energy impact from recommendation or estimate it
            impact = rec.get("energy_impact", self._estimate_single_recommendation_impact(rec))
            total_impact += impact
            
        return total_impact

    async def _store_optimization_results(
        self,
        zone_id: str,
        user_id: str,
        metrics: ComfortMetrics,
        recommendations: List[Dict[str, Any]]
    ) -> None:
        """Store optimization results in database."""
        try:
            result = {
                "timestamp": datetime.now(),
                "zone_id": zone_id,
                "user_id": user_id,
                "comfort_metrics": metrics.__dict__,
                "recommendations": recommendations,
                "comfort_score": self._calculate_comfort_score(metrics)
            }
            
            await self.db.save_optimization_result(result)
            
        except Exception as e:
            # Log error but don't raise to avoid disrupting the optimization process
            logger.error(f"Failed to store optimization results: {str(e)}")

    def close(self):
        """Close service connections."""
        try:
            self.db.close()
            asyncio.create_task(self.weather.close())
            asyncio.create_task(self.groq.close())
        except Exception as e:
            logger.error(f"Error closing services: {str(e)}")
