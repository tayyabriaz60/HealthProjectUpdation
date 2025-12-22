"""
FastAPI application entry point.
Run this file with Uvicorn to start the server.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.core.config import settings, BASE_DIR
from app.api.endpoints import chat, food, analytics, voice
from app.db import init_db

# Create FastAPI application instance
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Add CORS middleware for mobile app support
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

