import os
from typing import Dict, Any
from fastapi import FastAPI, Depends, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import logging
import uvicorn
from dotenv import load_dotenv

from api.endpoints import temperature, optimization, monitoring
from services.astra_db_service import AstraDBService
from services.weather_service import WeatherService
from services.groq_slm_service import GroqSLMService
from real_time.real_time_processing import RealTimeProcessor
from utils.logger import setup_logger
from utils.config import load_config

# Load environment variables
load_dotenv()

# Initialize logger
logger = setup_logger('main_app')

# Load configuration
config = load_config('config/app_config.yaml')

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next):
        try:
            logger.info(f"Request: {request.method} {request.url}")
            response = await call_next(request)
            logger.info(f"Response: {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise

class Services:
    """Service container for dependency injection."""
    
    def __init__(self):
        self.db = AstraDBService()
        self.weather = WeatherService()
        self.groq = GroqSLMService()
        self.real_time = RealTimeProcessor(
            db_service=self.db,
            alert_config=config.get('alerts')
        )
        
    async def cleanup(self):
        """Cleanup service connections."""
        self.db.close()
        await self.weather.close()
        await self.groq.close()
        await self.real_time.close()

# Initialize FastAPI application
app = FastAPI(
    title="HVAC Optimization System",
    description="Advanced HVAC system optimization and monitoring API",
    version="1.0.0",
    docs_url=None,  # Custom docs setup
    redoc_url=None
)

# Initialize services
services = Services()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['cors']['allowed_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_services() -> Services:
    """Dependency for service injection."""
    return services

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with authentication."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Documentation",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Include routers with service dependency
app.include_router(
    temperature.router,
    dependencies=[Depends(get_services)]
)
app.include_router(
    optimization.router,
    dependencies=[Depends(get_services)]
)
app.include_router(
    monitoring.router,
    dependencies=[Depends(get_services)]
)

# WebSocket endpoint for real-time processing
@app.websocket("/ws/{client_id}/{system_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    system_id: str,
    services: Services = Depends(get_services)
):
    """WebSocket endpoint for real-time updates."""
    await services.real_time.handle_websocket(
        websocket,
        client_id,
        system_id
    )

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting HVAC Optimization System")
    try:
        # Verify service connections
        await services.weather.test_connection()
        await services.groq.test_connection()
        
        # Initialize database tables
        services.db._init_tables()
        
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("Shutting down HVAC Optimization System")
    try:
        await services.cleanup()
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check(services: Services = Depends(get_services)):
    """System health check."""
    try:
        # Check service health
        db_status = "healthy" if services.db.session else "unhealthy"
        weather_status = "healthy" if await services.weather.test_connection() else "unhealthy"
        groq_status = "healthy" if await services.groq.test_connection() else "unhealthy"
        
        return {
            "status": "ok",
            "version": "1.0.0",
            "services": {
                "database": db_status,
                "weather": weather_status,
                "groq": groq_status
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    # Load server configuration
    host = config.get('server', {}).get('host', '0.0.0.0')
    port = config.get('server', {}).get('port', 8000)
    
    # Run application
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        reload=config.get('development_mode', False)
    )
