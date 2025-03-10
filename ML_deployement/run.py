import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",  # Use string format instead of imported app
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        reload_includes=["*.py"],  # Only reload on Python file changes
        workers=1  # Use single worker for development
    )
