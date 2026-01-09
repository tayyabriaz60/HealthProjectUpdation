"""
FastAPI application entry point.
Run this file with Uvicorn to start the server.
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import traceback

from app.core.config import settings, BASE_DIR
from app.core.logging_config import logger
from app.api.endpoints import chat, food, analytics, voice
from app.db import init_db

# Create FastAPI application instance
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# --- GLOBAL EXCEPTION HANDLER ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for any unhandled exceptions."""
    error_msg = str(exc)
    stack_trace = traceback.format_exc()
    
    # Log the full error details for developers
    logger.error(f"UNHANDLED ERROR: {request.method} {request.url}")
    logger.error(f"Error Message: {error_msg}")
    logger.error(f"Stack Trace:\n{stack_trace}")
    
    # Return a professional, clean response to the user
    return JSONResponse(
        status_code=500,
        content={
            "error": "We encountered an unexpected error while processing your request.",
            "code": "INTERNAL_SERVER_ERROR",
            "technical_message": error_msg if settings.DEBUG else "Please try again later."
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Specific handler for ValueErrors (usually business logic or validation issues)."""
    logger.warning(f"VALUE ERROR: {str(exc)} | Path: {request.url.path}")
    return JSONResponse(
        status_code=400,
        content={
            "error": str(exc),
            "code": "BAD_REQUEST"
        }
    )

# --- MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.api_router)
app.include_router(food.api_router)
app.include_router(analytics.api_router)
app.include_router(voice.api_router)

# Serve media files (stored images) from /media
media_root = BASE_DIR / "media"
media_root.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(media_root)), name="media")


@app.on_event("startup")
async def on_startup() -> None:
    """Application startup hook."""
    await init_db()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Chatbot API",
        "version": settings.APP_VERSION
    }



@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

