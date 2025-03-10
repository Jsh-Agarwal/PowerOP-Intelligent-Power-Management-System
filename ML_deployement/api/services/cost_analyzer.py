from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional
from ..utils.error_handlers import APIError
from .astra_db_service import AstraDBService

logger = logging.getLogger(__name__)

class CostAnalyzer:
    def __init__(self):
        self.db = AstraDBService()
        self.energy_rate = 0.15  # Cost per kWh
        
    async def analyze_cost(
        self,
        system_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze system costs for time period."""
        try:
            await self.db.connect()  # Add connection
            
            if not start_time:
                start_time = datetime.utcnow() - timedelta(days=30)
            if not end_time:
                end_time = datetime.utcnow()

            metrics = await self.db.get_system_metrics(
                system_id=system_id,
                start_time=start_time,
                end_time=end_time
            )

            # Fixed: Handle metrics as list directly
            if not metrics:
                return {
                    "status": "success",
                    "message": "No data available for the specified period",
                    "system_id": system_id,
                    "period": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    }
                }

            analysis = self._calculate_costs(metrics)
            recommendations = self._generate_recommendations(analysis)

            return {
                "status": "success",
                "system_id": system_id,
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "analysis": analysis,
                "recommendations": recommendations
            }

        except Exception as e:
            raise APIError(
                status_code=500,
                detail=f"Cost analysis failed: {str(e)}",
                error_code="ANALYSIS_ERROR"
            )
        finally:
            await self.db.close()  # Ensure connection is closed

    def _calculate_costs(self, metrics: list) -> Dict[str, Any]:
        """Calculate detailed cost analysis from metrics."""
        total_energy = sum(m["energy_consumption"] for m in metrics)
        total_cost = total_energy * self.energy_rate
        
        # Calculate peak and off-peak usage
        peak_hours = [h for h in range(9, 17)]  # 9 AM to 5 PM
        
        # Fix timestamp handling - ensure it's a string first
        peak_usage = sum(
            m["energy_consumption"] 
            for m in metrics 
            if isinstance(m["timestamp"], datetime) or 
               datetime.fromisoformat(str(m["timestamp"])).hour in peak_hours
        )
        
        return {
            "total_energy_kwh": round(total_energy, 2),
            "total_cost": round(total_cost, 2),
            "average_daily_cost": round(total_cost / (len(metrics) / 24), 2),
            "peak_usage_kwh": round(peak_usage, 2),
            "peak_usage_cost": round(peak_usage * self.energy_rate, 2),
            "efficiency_score": self._calculate_efficiency(metrics)
        }

    def _calculate_efficiency(self, metrics: list) -> float:
        """Calculate system efficiency score."""
        if not metrics:
            return 0.0
            
        efficiency_scores = [m.get("efficiency", 0) for m in metrics]
        return round(sum(efficiency_scores) / len(efficiency_scores), 2)

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> list:
        """Generate cost-saving recommendations."""
        recommendations = []
        
        if analysis["peak_usage_kwh"] > (analysis["total_energy_kwh"] * 0.4):
            recommendations.append({
                "type": "peak_usage",
                "message": "Consider shifting load to off-peak hours",
                "potential_savings": "10-15%"
            })
            
        if analysis["efficiency_score"] < 0.8:
            recommendations.append({
                "type": "efficiency",
                "message": "System efficiency below target",
                "potential_savings": "5-10%"
            })
            
        return recommendations

    async def get_temperature_history(
        self,
        system_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get temperature history with analysis."""
        try:
            data = await self.db.get_temperature_data(
                device_id=system_id,  # Assuming device_id maps to system_id
                zone_id="main",  # Default zone
                start_time=start_time,
                end_time=end_time
            )

            if not data:
                return {
                    "status": "success",
                    "message": "No data available for the specified period",
                    "data": []
                }

            analysis = self._analyze_temperature_data(data)
            
            return {
                "status": "success",
                "system_id": system_id,
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "data": data,
                "analysis": analysis
            }

        except Exception as e:
            raise APIError(
                status_code=500,
                detail=f"Temperature analysis failed: {str(e)}",
                error_code="ANALYSIS_ERROR"
            )

    def _analyze_temperature_data(self, data: list) -> Dict[str, Any]:
        """Analyze temperature patterns."""
        temperatures = [d["temperature"] for d in data]
        return {
            "average": round(sum(temperatures) / len(temperatures), 2),
            "min": round(min(temperatures), 2),
            "max": round(max(temperatures), 2),
            "variance": round(sum((t - sum(temperatures)/len(temperatures))**2 for t in temperatures) / len(temperatures), 2)
        }
