import os
import logging
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from dotenv import load_dotenv

from routers import api_router

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

# Add global exception handler for validation errors
@app.exception_handler(RequestValidationError)
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: Exception):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request format. Make sure all required fields are present and properly formatted.",
            "error_details": str(exc)
        }
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