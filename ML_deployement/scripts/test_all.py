import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from utils.logger import setup_detailed_logger, log_request_response
import traceback
from typing import Dict, Any, Optional

logger = setup_detailed_logger("api_test")
BASE_URL = "http://localhost:8000"

async def make_request(
    session: aiohttp.ClientSession,
    method: str,
    endpoint: str,
    headers: Dict = None,
    params: Dict = None,
    json_data: Dict = None,
    form_data: Dict = None
) -> Optional[Dict]:
    """Make HTTP request with detailed logging."""
    url = f"{BASE_URL}{endpoint}"
    try:
        kwargs = {
            "headers": headers,
            "params": params,
        }
        if json_data:
            kwargs["json"] = json_data
        if form_data:
            kwargs["data"] = form_data

        async with getattr(session, method.lower())(url, **kwargs) as response:
            response_data = await response.json()
            log_request_response(
                logger,
                method,
                url,
                request_data={"params": params, "json": json_data, "form": form_data},
                response_data=response_data,
                status_code=response.status
            )
            return response_data
    except Exception as e:
        logger.error(
            f"Request failed: {method} {url}\n"
            f"Error: {str(e)}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        return None

async def get_token(session: aiohttp.ClientSession) -> str:
    """Get authentication token."""
    form_data = {
        "grant_type": "password",  # Add this
        "username": "test",
        "password": "test"
    }
    token_data = await make_request(
        session, 
        "POST", 
        "/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        form_data=form_data
    )
    
    if not token_data or "access_token" not in token_data:
        raise ValueError(f"Failed to get token: {token_data}")
    return token_data["access_token"]

async def test_endpoints():
    """Test all API endpoints with retry logic."""
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Get token
            token = await get_token(session)
            headers = {"Authorization": f"Bearer {token}"}
            
            # Define comprehensive test cases
            test_cases = [
                # Weather endpoints
                ("GET", "/api/weather/current", {"location": "London"}),
                ("GET", "/api/weather/forecast", {"location": "London", "days": 5}),
                
                # Temperature endpoints
                ("POST", "/api/temperature/predict", {
                    "device_id": "test_device",
                    "zone_id": "test_zone",
                    "timestamps": [(datetime.now() + timedelta(hours=i)).isoformat() 
                                 for i in range(24)],
                    "features": {
                        "temperature": [22.0] * 24,
                        "humidity": [50.0] * 24,
                        "power": [1000.0] * 24
                    }
                }),
                ("POST", "/api/temperature/batch", {
                    "requests": [
                        {
                            "device_id": f"device_{i}",
                            "zone_id": "zone_1",
                            "timestamps": [(datetime.now() + timedelta(hours=h)).isoformat() 
                                         for h in range(24)],
                            "features": {
                                "temperature": [22.0] * 24,
                                "humidity": [50.0] * 24,
                                "power": [1000.0] * 24
                            }
                        } for i in range(3)
                    ]
                }),
                ("GET", "/api/temperature/history", {
                    "device_id": "test_device",
                    "zone_id": "test_zone",
                    "start_time": (datetime.now() - timedelta(days=1)).isoformat()
                }),
                ("GET", "/api/temperature/current", {
                    "device_id": "test_device",
                    "zone_id": "test_zone"
                }),
                
                # System status endpoints
                ("GET", "/api/status/system/test_system", None),
                ("GET", "/api/health", None),
                ("GET", "/api/status/metrics", {
                    "system_id": "test_system",
                    "start_time": (datetime.now() - timedelta(hours=24)).isoformat(),
                    "end_time": datetime.now().isoformat()
                }),
                
                # Optimization endpoints
                ("POST", "/api/optimize/system", {
                    "system_id": "test_system",
                    "target_metric": "efficiency",
                    "constraints": {
                        "min_temperature": 20.0,
                        "max_temperature": 26.0
                    },
                    "current_state": {
                        "temperature": 23.5,
                        "humidity": 55.0,
                        "power": 1200.0
                    }
                }),
                ("POST", "/api/optimize/comfort", {
                    "system_id": "test_system",
                    "target_metric": "comfort",
                    "constraints": {"min_temperature": 21.0},
                    "current_state": {"temperature": 24.0}
                }),
                ("POST", "/api/optimize/energy", {
                    "system_id": "test_system",
                    "target_metric": "energy",
                    "constraints": {
                        "max_power": 1500.0
                    },
                    "current_state": {
                        "power": 1200.0,
                        "temperature": 23.0
                    }
                }),
                ("POST", "/api/optimize/schedule", {
                    "system_id": "test_system",
                    "current_state": {
                        "temperature": 23.0,
                        "humidity": 50.0,
                        "power": 1000.0
                    },
                    "start_time": datetime.now().isoformat(),
                    "end_time": (datetime.now() + timedelta(days=1)).isoformat(),
                    "interval": 60
                }),
                
                # Database operations
                ("POST", "/astra/create_table", {
                    "table_name": "test_temperatures",
                    "schema": """
                        CREATE TABLE IF NOT EXISTS test_temperatures (
                            device_id text,
                            timestamp timestamp,
                            temperature double,
                            humidity double,
                            PRIMARY KEY (device_id, timestamp)
                        )
                    """
                }),
                
                # Groq SLM endpoints
                ("GET", "/groq/context/test_context", None),
                ("POST", "/groq/optimize", {
                    "prompt": "Optimize HVAC system for energy efficiency",
                    "context": {
                        "current_temperature": 24.0,
                        "target_temperature": 22.0,
                        "power_consumption": 1200.0
                    }
                })
            ]
            
            # Test each endpoint with retries
            results = {}
            for method, endpoint, data in test_cases:
                logger.info(f"\nTesting endpoint: {method} {endpoint}")
                result = await test_endpoint_with_retry(
                    session, method, endpoint, headers, data
                )
                results[endpoint] = {
                    "success": result is not None,
                    "response": result,
                    "status": "Pass" if result is not None else "Fail"
                }
            
            # Log detailed test summary
            logger.info("\nTest Summary:")
            for endpoint, result in results.items():
                status_symbol = "+" if result["success"] else "x"
                logger.info(f"{status_symbol} {result['status']} - {endpoint}")
                if not result["success"]:
                    logger.error(f"  Response: {result['response']}")

            return results

    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}\n{traceback.format_exc()}")
        raise

async def test_endpoint_with_retry(
    session: aiohttp.ClientSession,
    method: str,
    endpoint: str,
    headers: Dict,
    data: Dict,
    max_retries: int = 3
) -> Optional[Dict]:
    """Test endpoint with retry logic."""
    last_error = None
    for attempt in range(max_retries):
        try:
            result = await make_request(
                session, 
                method, 
                endpoint,
                headers=headers,
                json_data=data if method == "POST" else None,
                params=data if method == "GET" else None
            )
            if result:
                return result
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
                await asyncio.sleep(1)
    
    logger.error(f"All retries failed for {endpoint}: {str(last_error)}")
    return None

def format_test_result(success: bool, endpoint: str) -> str:
    """Format test result with ASCII symbols."""
    symbol = "+" if success else "x"
    return f"{symbol} {'Pass' if success else 'Fail'} - {endpoint}"

if __name__ == "__main__":
    logger.info("Starting comprehensive API test...")
    try:
        asyncio.run(test_endpoints())
        logger.info("API test completed successfully!")
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
