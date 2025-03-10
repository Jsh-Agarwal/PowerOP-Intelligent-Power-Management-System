from typing import Dict, Any, List
import json

class GroqPromptTemplates:
    """Templates for generating Groq SLM prompts."""
    
    @staticmethod
    def optimization_system_prompt() -> str:
        """System prompt for optimization."""
        return """You are an expert HVAC systems engineer specializing in optimization.
Your task is to provide specific, actionable recommendations to optimize HVAC operations
based on the provided system data and weather conditions.

FORMAT YOUR RESPONSE IN JSON with the following structure:
{
    "recommendations": [
        {
            "title": "Short descriptive title",
            "action": "Specific action to take",
            "rationale": "Why this action is recommended",
            "expected_benefit": "Quantitative or qualitative benefit",
            "priority": "high|medium|low"
        }
    ],
    "summary": "Brief summary of overall optimization approach"
}
"""

    @staticmethod
    def optimization_prompt(
        hvac_data: Dict[str, Any],
        weather_data: Dict[str, Any],
        target: str
    ) -> str:
        """Generate optimization prompt."""
        hvac_json = json.dumps(hvac_data, indent=2)
        weather_json = json.dumps(weather_data, indent=2)
        
        return f"""I need to optimize an HVAC system with the following target: {target.upper()}.

HVAC SYSTEM DATA:
{hvac_json}

WEATHER FORECAST:
{weather_json}

Based on this information, provide specific recommendations to optimize the HVAC system.
Focus on {target} while maintaining adequate performance in other areas.
Provide 3-5 actionable recommendations with clear rationale and expected benefits.
"""

    @staticmethod
    def anomaly_system_prompt() -> str:
        """System prompt for anomaly analysis."""
        return """You are an expert HVAC diagnostic technician specializing in fault detection.
Your task is to analyze anomalies detected in an HVAC system and provide diagnosis and recommendations.

FORMAT YOUR RESPONSE IN JSON with the following structure:
{
    "diagnosis": {
        "primary_cause": "Most likely cause of the anomaly",
        "confidence": "high|medium|low",
        "alternative_causes": ["Other possible causes"],
        "severity": "critical|high|medium|low",
        "impact": "Description of potential impact if not addressed"
    },
    "recommendations": [
        {
            "action": "Specific action to take",
            "urgency": "immediate|soon|scheduled",
            "expertise_required": "technician|engineer|specialist"
        }
    ],
    "additional_diagnostics": ["Any additional tests recommended"]
}
"""

    @staticmethod
    def anomaly_prompt(
        anomaly_data: Dict[str, Any],
        system_context: Dict[str, Any]
    ) -> str:
        """Generate anomaly analysis prompt."""
        anomaly_json = json.dumps(anomaly_data, indent=2)
        context_json = json.dumps(system_context, indent=2)
        
        return f"""I need to analyze anomalies detected in an HVAC system.

ANOMALY DATA:
{anomaly_json}

SYSTEM CONTEXT:
{context_json}

Based on this information, provide a detailed diagnosis of the anomalies detected.
Identify the most likely causes, assess the severity, and recommend specific actions.
Include any additional diagnostic tests that should be performed to confirm the diagnosis.
"""
