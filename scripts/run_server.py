"""
Main script to run the API server
"""
import uvicorn
from src.api.server import app
from config.settings import settings

if __name__ == "__main__":
    print("Starting JobKorea AI Challenge API Server...")
    print(f"Server will be available at http://{settings.host}:{settings.port}")
    print("API documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )