"""
Settings and Configuration variables.
Handles API keys, model names, and other configuration settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path

# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Gemini API Configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    
    # Application Configuration
    APP_NAME: str = "Chatbot API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: Optional[str] = None
    
    # System Prompt for Diabetes Health Assistant
    SYSTEM_PROMPT: str = (
        "You are an ultra-concise, professional diabetes health assistant. "
        "Goal: Provide safe, actionable advice using latest health data. "
        "Rules for extreme brevity (Token Efficiency):\n"
        "- LIMIT: Maximum 2 sentences or 40 words per response.\n"
        "- NO FILLER: Skip 'I understand', 'Sure', 'Hello', or 'Here is your advice'.\n"
        "- DIRECT: Start answering the question immediately.\n"
        "- PLAIN TEXT: No markdown (**bold**, *italics*, # headers).\n"
        "- IMAGE AWARENESS: Use provided image descriptions. Never say you can't see images.\n"
        "- SAFETY: Never recommend medication changes. If glucose <70, give 15g fast carbs advice immediately.\n"
        "- MULTILINGUAL: Detect and respond in the SAME language the user speaks (Urdu, English, Arabic, etc.). "
        "If user speaks Urdu, respond in Urdu. If English, respond in English. Match their language exactly. "
        "DO NOT respond in Hindi - always prefer Urdu over Hindi.\n"
        "- Focus strictly on current data context. Be empathetic but very brief."
    )
    
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Create a global settings instance
settings = Settings()

