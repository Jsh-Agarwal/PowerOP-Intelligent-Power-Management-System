import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiohttp
from dotenv import load_dotenv
from utils.exceptions import HVACSystemError  # Use absolute import

logger = logging.getLogger('groq_service')

class GroqServiceError(HVACSystemError):
    """Base exception for Groq SLM service errors."""
    pass

class GroqAPIError(GroqServiceError):
    """Raised when Groq API request fails."""
    def __init__(self, message: str, response: Optional[Dict] = None):
        self.response = response
        super().__init__(f"Groq API error: {message}")

class GroqAuthError(GroqServiceError):
    """Raised when authentication fails."""
    def __init__(self):
        super().__init__("Groq authentication failed")

class GroqSLMService:
    """Service for interacting with Groq SLM API."""
    
    def __init__(self):
        """Initialize Groq service."""
        load_dotenv()
        self.api_key = os.getenv('GROQ_SLM_API_KEY')
        if not self.api_key:
            raise GroqAuthError()
            
        self.base_url = "https://api.groq.com/v1"
        self.session = None
        self.default_params = {
            "temperature": 0.7,
            "max_tokens": 1024
        }

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self.session

    async def generate_hvac_optimization(
        self,
        hvac_data: Dict[str, Any],
        weather_data: Optional[Dict[str, Any]] = None,
        optimization_target: str = "efficiency",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate HVAC optimization recommendations using Groq LLM.
        
        Args:
            hvac_data: Current HVAC system data
            weather_data: Optional weather forecast data
            optimization_target: Target for optimization (efficiency/comfort/etc)
            
        Returns:
            Dictionary containing recommendations and metrics
        """
        try:
            # Prepare context
            context = self._prepare_context(
                hvac_data,
                weather_data,
                optimization_target
            )
            
            # Get API response
            session = await self.get_session()
            async with session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": "mixtral-8x7b-32768",
                    "messages": [{"role": "user", "content": context}],
                    **self.default_params,
                    **kwargs
                }
            ) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise GroqAPIError(
                        "Failed to get optimization recommendations",
                        error_data
                    )
                    
                data = await response.json()
                return self._process_response(data)
                
        except Exception as e:
            raise GroqServiceError(f"Optimization generation failed: {str(e)}")

    def _prepare_context(
        self,
        hvac_data: Dict[str, Any],
        weather_data: Optional[Dict[str, Any]],
        optimization_target: str
    ) -> str:
        """Prepare context for LLM prompt."""
        context = [
            "Based on the following HVAC system data:",
            json.dumps(hvac_data, indent=2)
        ]
        
        if weather_data:
            context.extend([
                "\nAnd weather forecast:",
                json.dumps(weather_data, indent=2)
            ])
            
        context.extend([
            f"\nProvide optimization recommendations for {optimization_target}.",
            "Include specific setpoint adjustments and expected savings."
        ])
        
        return "\n".join(context)

    def _process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure LLM response."""
        try:
            content = response['choices'][0]['message']['content']
            
            # Parse recommendations from response
            # This is a simplified example - actual parsing would be more robust
            recommendations = []
            expected_savings = 0.0
            confidence_score = 0.85  # Example confidence score
            
            # Add parsed data
            return {
                "recommendations": recommendations,
                "expected_savings": expected_savings,
                "confidence_score": confidence_score,
                "raw_response": content
            }
            
        except Exception as e:
            raise GroqServiceError(f"Failed to process response: {str(e)}")

    async def test_connection(self) -> bool:
        """Test API connection."""
        try:
            session = await self.get_session()
            async with session.get(f"{self.base_url}/models") as response:
                return response.status == 200
        except Exception:
            return False

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def __del__(self):
        """Cleanup on deletion."""
        if self.session and not self.session.closed:
            import asyncio
            asyncio.create_task(self.session.close())
