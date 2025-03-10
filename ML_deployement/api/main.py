from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
import uvicorn
from pathlib import Path
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta, datetime
import logging
from fastapi.responses import JSONResponse
import traceback
import asyncio
from contextlib import AsyncExitStack
from typing import Callable

# Import auth module
from .auth import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, oauth2_scheme

# Import middlewares module
from .middlewares import log_requests_middleware

# Import error handlers
from .utils.error_handlers import APIError

# Import logging configuration
from .utils.logging_config import setup_logging

# Setup logging before creating FastAPI app
setup_logging()
logger = logging.getLogger("api")

# Update app description
app = FastAPI(
    title="HVAC Optimization API",
    description="""
    API for HVAC system optimization and monitoring.
    
    New Features:
    - Anomaly Detection using AutoEncoder
    - Daily Temperature Analysis
    - Cost Analysis
    - LLM-based System Analysis
    
    All endpoints require authentication using Bearer token.
    """,
    version="1.1.0",
    docs_url=None,  # Disable default docs
    redoc_url=None  # Disable default redoc
)

# Mount static files directory
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
app.middleware("http")(log_requests_middleware)

# Serve Swagger UI files
swagger_ui_files = {
    "swagger-ui-bundle.js": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
    "swagger-ui.css": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"
}

# Custom docs endpoint with CDN resources
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="HVAC Optimization API - Documentation",
        swagger_js_url=swagger_ui_files["swagger-ui-bundle.js"],
        swagger_css_url=swagger_ui_files["swagger-ui.css"],
    )

# Import all routers
from .endpoints import temperature, optimization, monitoring, weather

# Register all routers
app.include_router(temperature.router)
app.include_router(optimization.router)
app.include_router(monitoring.router)
app.include_router(weather.router)

# Import additional routers
from .endpoints import groq, astra, health

# Register additional routers
app.include_router(groq.router)
app.include_router(astra.router)

# Import new routers
from .endpoints import control, analysis

# Register new routers
app.include_router(control.router)
app.include_router(analysis.router)

# Remove the duplicate health endpoint from main.py and use the one from health.py
app.include_router(health.router)

@app.post("/token", tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Get authentication token."""
    try:
        # For testing purposes, accept any credentials
        if not form_data.username or not form_data.password:
            raise HTTPException(status_code=422, detail="Invalid credentials")
            
        access_token = create_access_token(
            data={"sub": form_data.username},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "access_token": access_token,
                "token_type": "bearer"
            }
        )
    except Exception as e:
        logger.error(f"Token generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.middleware("http")
async def timeout_middleware(request: Request, call_next: Callable):
    try:
        async with AsyncExitStack() as stack:
            if hasattr(asyncio, 'timeout'):
                await stack.enter_async_context(asyncio.timeout(30))
            else:
                # Fallback for Python < 3.11
                await stack.enter_async_context(asyncio.TimeoutError(30))
            response = await call_next(request)
            return response
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"detail": "Request timeout"}
        )

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """Handle API errors with detailed responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "detail": exc.detail,
            "path": str(request.url),
            "method": request.method
        }
    )

@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Handle any unhandled exceptions"""
    logger.error(f"Unhandled error: {str(exc)}")
    logger.debug(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": {
                "message": str(exc),
                "error_code": exc.__class__.__name__,
                "traceback": traceback.format_exc(),
                "path": str(request.url),
                "method": request.method
            }
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.get("/", tags=["System"])
async def root():
    """API root endpoint."""
    return {
        "name": "HVAC Optimization API",
        "version": "1.0.0", 
        "status": "running",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
