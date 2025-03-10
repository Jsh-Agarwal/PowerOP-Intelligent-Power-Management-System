import requests
import json
from datetime import datetime
import logging
from pprint import pformat

BASE_URL = "http://localhost:8000"
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def log_request(method: str, url: str, **kwargs):
    """Log request details."""
    logger.debug(f"\n{'='*50}")
    logger.debug(f"REQUEST: {method} {url}")
    if 'headers' in kwargs:
        logger.debug(f"Headers: {pformat(kwargs['headers'])}")
    if 'params' in kwargs:
        logger.debug(f"Params: {pformat(kwargs['params'])}")
    if 'json' in kwargs:
        logger.debug(f"Body: {pformat(kwargs['json'])}")

def log_response(response: requests.Response):
    """Log response details."""
    logger.debug(f"\nRESPONSE: {response.status_code}")
    try:
        logger.debug(f"Body: {pformat(response.json())}")
    except:
        logger.debug(f"Body: {response.text}")
    logger.debug(f"{'='*50}\n")

def get_token():
    """Get authentication token."""
    url = f"{BASE_URL}/token"
    data = {"username": "test", "password": "test"}
    
    logger.info("Getting authentication token...")
    log_request("POST", url, data=data)
    
    response = requests.post(url, data=data)
    log_response(response)
    
    return response.json()["access_token"]

def test_api():
    """Test main API endpoints."""
    # Get token
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # 1. Test weather endpoint
    logger.info("\nTesting weather endpoint...")
    weather_url = f"{BASE_URL}/api/weather/current"
    params = {"location": "London"}
    
    log_request("GET", weather_url, headers=headers, params=params)
    weather = requests.get(weather_url, params=params, headers=headers)
    log_response(weather)
    
    # 2. Test temperature prediction
    logger.info("\nTesting temperature prediction...")
    pred_url = f"{BASE_URL}/api/temperature/predict"
    pred_data = {
        "device_id": "test_device",
        "zone_id": "test_zone",
        "timestamps": [datetime.now().isoformat()],
        "features": {
            "temperature": [22.0],
            "humidity": [50.0],
            "power": [1000.0]
        }
    }
    
    log_request("POST", pred_url, headers=headers, json=pred_data)
    prediction = requests.post(pred_url, json=pred_data, headers=headers)
    log_response(prediction)
    
    # 3. Test system status
    logger.info("\nTesting system status...")
    status_url = f"{BASE_URL}/api/status/system/test_system"
    
    log_request("GET", status_url, headers=headers)
    status = requests.get(status_url, headers=headers)
    log_response(status)

if __name__ == "__main__":
    test_api()
