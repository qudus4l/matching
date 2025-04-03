import os
import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.routers import api_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("unified_api.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting Unified Matching API")

# Initialize FastAPI application
app = FastAPI(
    title="Unified Matching API",
    description="API for job description generation and AI-based profile matching",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include routers
app.include_router(api_router.router)
logger.info("API router initialized")

# Root endpoint
@app.get("/")
async def root():
    """Return welcome message."""
    logger.info("Root endpoint accessed")
    return {
        "message": "Welcome to the Unified Matching API",
        "description": "This API provides job description generation and AI-based profile matching",
        "documentation": "/docs",
        "version": "1.0.0"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Return API health status."""
    logger.info("Health check endpoint accessed")
    return {"status": "OK", "service": "Unified Matching API"} 