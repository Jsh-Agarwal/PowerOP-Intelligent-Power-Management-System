import sys
from pathlib import Path
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
import logging
from api.utils.logging_config import setup_logging
from test_data.test_payloads import (
    TEMPERATURE_PREDICTION,
    BATCH_PREDICTION,
    SYSTEM_OPTIMIZATION,
    SCHEDULE_OPTIMIZATION,
    TEMPERATURE_CONTROL,
    POWER_CONTROL,
    ANOMALY_DETECTION,
    LLM_ANALYSIS,
    TEMPERATURE_QUERY,  # Added import
    COMFORT_OPTIMIZATION,  # Added import
    ENERGY_OPTIMIZATION  # Added import
)

# Add project root to Python path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

# Setup logging with minimal console output
log_file = setup_logging(
    console_level=logging.WARNING,  # Only show warnings and errors in console
    file_level=logging.DEBUG  # Keep detailed logs in file
)

logger = logging.getLogger(__name__)

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)

# Base URL for the deployed API
#BASE_URL = "http://hvacapi.b2a6gddyhrfvcpb6.westindia.azurecontainer.io:8000"
BASE_URL = "http://localhost:8000"

class TestResults:
    """Simple class to track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0
        self.results = []  # Store all results
        self.failed_tests = []
        self.detailed_responses = []  # Add this to store API responses

    def add_result(self, name: str, status: int, success: bool, error=None, request=None, response=None):
        self.total += 1
        result = {
            "name": name,
            "status": status,
            "success": success,
            "error": error
        }
        self.results.append(result)
        
        if success:
            self.passed += 1
            print(f"[PASS] {name}: {status}")
        else:
            self.failed += 1
            self.failed_tests.append(result)
            print(f"[FAIL] {name}: {status} - {error if error else ''}")

        # Add detailed response data
        self.detailed_responses.append({
            "test_name": name,
            "request": {
                "url": request.get("url"),
                "method": request.get("method"),
                "params": request.get("params"),
                "payload": request.get("json")
            },
            "response": {
                "status": status,
                "body": response,
                "success": success,
                "error": error
            },
            "timestamp": datetime.utcnow().isoformat()
        })

    def print_summary(self):
        print("\nTest Results Summary")
        print("===================")
        print(f"Total Tests:  {self.total}")
        print(f"Passed:       {self.passed}")
        print(f"Failed:       {self.failed}")
        
        if self.failed_tests:
            print("\nFailed Tests:")
            print("------------")
            for test in self.failed_tests:
                error_msg = test.get('error', 'No error details')
                print(f"[FAIL] {test['name']}: {test['status']} - {error_msg}")

    def __len__(self):
        return self.total

    def __iter__(self):
        return iter(self.results)

async def run_api_tests():
    """Run tests for all API endpoints."""
    logger.info(f"Starting API tests. Detailed logs will be written to: {log_file}")
    
    results = TestResults()
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test authentication first
            try:
                logger.info("Testing authentication...")
                token_response = await session.post(
                    f"{BASE_URL}/token",
                    data={
                        "username": "test",
                        "password": "test"
                    }
                )
                token_text = await token_response.text()
                logger.debug(f"Auth response status: {token_response.status}", extra={
                    "data": {
                        "status": token_response.status,
                        "body": token_text
                    }
                })
                
                if token_response.status != 200:
                    logger.error(f"Authentication failed: {token_text}")
                    return []
                
                token_data = await token_response.json()
                headers = {
                    "Authorization": f"Bearer {token_data['access_token']}"
                }
            except Exception as e:
                logger.error(f"Authentication error: {str(e)}", exc_info=True)
                return []

            # Test endpoints
            test_cases = [
                # System endpoints
                {
                    "name": "Root",
                    "method": "GET",
                    "url": f"{BASE_URL}/"
                },
                {
                    "name": "Health Check",
                    "method": "GET",
                    "url": f"{BASE_URL}/api/health"
                },
                {
                    "name": "Root Health Check",
                    "method": "GET",
                    "url": f"{BASE_URL}/health"
                },

                # Temperature Management
                {
                    "name": "Predict Temperature",
                    "method": "POST",
                    "url": f"{BASE_URL}/api/temperature/predict",
                    "json": TEMPERATURE_PREDICTION
                },
                {
                    "name": "Train Temperature Model",
                    "method": "POST",
                    "url": f"{BASE_URL}/api/temperature/train",
                    "json": {
                        "system_id": "test_system"
                    }
                },
                {
                    "name": "Get Temperature History",
                    "method": "GET",
                    "url": f"{BASE_URL}/api/temperature/history",
                    "params": TEMPERATURE_QUERY  # Added query parameters
                },
                {
                    "name": "Get Current Temperature",
                    "method": "GET",
                    "url": f"{BASE_URL}/api/temperature/current",
                    "params": TEMPERATURE_QUERY  # Added query parameters
                },
                {
                    "name": "Batch Predict Temperature",
                    "method": "POST",
                    "url": f"{BASE_URL}/api/temperature/batch",
                    "json": BATCH_PREDICTION
                },

                # System Optimization
                {
                    "name": "Optimize System",
                    "method": "POST",
                    "url": f"{BASE_URL}/api/optimize/system",
                    "json": SYSTEM_OPTIMIZATION
                },
                {
                    "name": "Optimize Comfort",
                    "method": "POST",
                    "url": f"{BASE_URL}/api/optimize/comfort",
                    "json": COMFORT_OPTIMIZATION  # Updated payload
                },
                {
                    "name": "Optimize Energy",
                    "method": "POST",
                    "url": f"{BASE_URL}/api/optimize/energy",
                    "json": ENERGY_OPTIMIZATION  # Updated payload
                },
                {
                    "name": "Optimize Schedule",
                    "method": "POST",
                    "url": f"{BASE_URL}/api/optimize/schedule",
                    "json": SCHEDULE_OPTIMIZATION
                },

                # System Monitoring
                {
                    "name": "Get System Status",
                    "method": "GET",
                    "url": f"{BASE_URL}/api/status/system/test_system"
                },
                {
                    "name": "Get System Metrics",
                    "method": "GET",
                    "url": f"{BASE_URL}/api/status/metrics",
                    "params": {"system_id": "test_system"}
                },

                # Weather Data
                {
                    "name": "Get Current Weather",
                    "method": "GET",
                    "url": f"{BASE_URL}/api/weather/current",
                    "params": {"location": "test_location"}
                },
                {
                    "name": "Get Weather Forecast",
                    "method": "GET",
                    "url": f"{BASE_URL}/api/weather/forecast",
                    "params": {"location": "test_location"}
                },

                # Groq LLM Service
                {
                    "name": "Get Context",
                    "method": "GET",
                    "url": f"{BASE_URL}/groq/context/test_context"
                },
                {
                    "name": "Optimize With Groq",
                    "method": "POST",
                    "url": f"{BASE_URL}/groq/optimize",
                    "json": {
                        "query": "Optimize HVAC efficiency",
                        "context": {"system_id": "test_system"}
                    }
                },

                # AstraDB Management
                {
                    "name": "Create Table",
                    "method": "POST",
                    "url": f"{BASE_URL}/astra/create_table",
                    "json": {
                        "table_name": "test_table",
                        "schema": {"id": "uuid", "name": "text"}
                    }
                },

                # System Control
                {
                    "name": "Set Temperature",
                    "method": "POST",
                    "url": f"{BASE_URL}/api/control/temperature",
                    "json": TEMPERATURE_CONTROL
                },
                {
                    "name": "Set Power State",
                    "method": "POST",
                    "url": f"{BASE_URL}/api/control/power",
                    "json": POWER_CONTROL
                },
                {
                    "name": "Increment Temperature",
                    "method": "POST",
                    "url": f"{BASE_URL}/api/control/temperature/increment/test_system"
                },
                {
                    "name": "Decrement Temperature",
                    "method": "POST",
                    "url": f"{BASE_URL}/api/control/temperature/decrement/test_system"
                },

                # System Analysis
                {
                    "name": "Daily Temperature Analysis",
                    "method": "GET",
                    "url": f"{BASE_URL}/api/analysis/temperature/daily/test_system",
                    "params": {
                        "date": datetime.now().date().isoformat()
                    }
                },
                {
                    "name": "Cost Analysis",
                    "method": "GET",
                    "url": f"{BASE_URL}/api/analysis/cost/test_system",
                    "params": {
                        "start_time": (datetime.now() - timedelta(days=7)).isoformat(),
                        "end_time": datetime.now().isoformat()
                    }
                },
                {
                    "name": "LLM Analysis",
                    "method": "POST",
                    "url": f"{BASE_URL}/api/analysis/optimize/llm/test_system",
                    "json": {
                        "query": "Analyze system efficiency",
                        "context": {
                            "temperature": 23.5,
                            "power": 1000.0,
                            "runtime_hours": 24
                        }
                    }
                },
                {
                    "name": "Anomaly Detection",
                    "method": "POST",
                    "url": f"{BASE_URL}/api/analysis/anomaly/detect/test_system",
                    "json": ANOMALY_DETECTION
                }
            ]

            for test in test_cases:
                try:
                    print(f"\n=== Testing: {test['name']} ===")
                    print(f"Request: {test['url']}")
                    if test.get('json'):
                        print(f"Payload: {json.dumps(test['json'], indent=2)}")
                    if test.get('params'):
                        print(f"Params: {test['params']}")
                    
                    if test["method"] == "GET":
                        response = await session.get(
                            test["url"],
                            headers=headers,
                            params=test.get("params")
                        )
                    else:
                        response = await session.post(
                            test["url"],
                            headers=headers,
                            json=test.get("json")
                        )

                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                        print(f"Response ({response.status}):")
                        print(json.dumps(response_json, indent=2))
                    except:
                        print(f"Raw Response ({response.status}):")
                        print(response_text)

                    success = response.status in [200, 201, 207]
                    error = response_json.get('detail') if not success else None
                    
                    results.add_result(
                        name=test["name"],
                        status=response.status,
                        success=success,
                        error=error,
                        request=test,
                        response=response_json
                    )

                except Exception as e:
                    print(f"Error: {str(e)}")
                    results.add_result(
                        name=test["name"],
                        status="error",
                        success=False,
                        error=str(e),
                        request=test,
                        response=None
                    )

            # Save results summary to the log
            logger.info("\nTest Results Summary", extra={
                "data": {
                    "total_tests": len(results),
                    "successful": sum(1 for r in results if r["success"]),
                    "failed": sum(1 for r in results if not r["success"])
                }
            })
            
            for result in results:
                status_symbol = "[PASS]" if result["success"] else "[FAIL]"
                logger.info(f"{status_symbol} {result['name']}: {result['status']}")
                if not result["success"]:
                    logger.debug(f"Failed test details: {result}")

            # Save detailed results to file
            result_data = {
                "summary": {
                    "total_tests": len(results),
                    "passed": results.passed,
                    "failed": results.failed,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "failed_tests": results.failed_tests,
                "detailed_responses": results.detailed_responses
            }
            
            with open(TEST_DATA_DIR / "test_results.json", "w") as f:
                json.dump(result_data, f, indent=2)

            # Print final summary
            results.print_summary()

            return results.failed == 0
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        success = asyncio.run(run_api_tests())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
