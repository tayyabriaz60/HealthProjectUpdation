from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import os
import base64
import tempfile
import shutil

from app.services.audio_service import AudioService
from app.services.gemini_service import GeminiService
from app.services.health_service import HealthService
from app.db import get_db
from app.models import ChatSession, Message
from app.core.logging_config import logger

# Router for voice endpoints
api_router = APIRouter(prefix="/api/voice", tags=["voice"])

# Lazy initialization
_audio_service_instance = None
_gemini_service_instance = None

def get_audio_service() -> AudioService:
    global _audio_service_instance
    if _audio_service_instance is None:
        _audio_service_instance = AudioService()
    return _audio_service_instance

def get_gemini_service() -> GeminiService:
    global _gemini_service_instance
    if _gemini_service_instance is None:
        _gemini_service_instance = GeminiService()
    return _gemini_service_instance

@api_router.post("/chat")
async def voice_chat(
    file: UploadFile = File(...),
    chat_id: str = Query(None, description="Optional chat session ID"),
    user_id: str = Query(None, description="Optional user ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts an audio file, uses Gemini 2.5 Flash for native transcription and 
    personalized response, then converts to audio with Edge-TTS.
    """
    temp_output_path = None

    try:
        audio_service = get_audio_service()
        gemini_service = get_gemini_service()
        
        logger.info(f"Voice Chat Request (Gemini Native) - User: {user_id} | Session: {chat_id}")

        # 1. Read audio bytes
        audio_data = await file.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        # Detect MIME type from filename or content_type
        mime_type = file.content_type or "audio/wav"
        
        # Fix: If content_type is generic, detect from filename
        if mime_type == "application/octet-stream" or not mime_type:
            filename = file.filename or ""
            filename_lower = filename.lower()
            
            if filename_lower.endswith('.wav'):
                mime_type = "audio/wav"
            elif filename_lower.endswith('.mp3'):
                mime_type = "audio/mpeg"
            elif filename_lower.endswith('.m4a'):
                mime_type = "audio/mp4"
            elif filename_lower.endswith('.ogg'):
                mime_type = "audio/ogg"
            elif filename_lower.endswith('.webm'):
                mime_type = "audio/webm"
            else:
                # Default to WAV (Flutter uses pcm16WAV)
                mime_type = "audio/wav"
        
        logger.info(f"Detected audio MIME type: {mime_type} (filename: {file.filename})")

        # 2. Ensure Session & Fetch Context
        current_chat_id = await gemini_service.ensure_session(chat_id, db)
        
        health_context = ""
        if user_id:
            health_context = await HealthService.get_patient_context_string(db, user_id)
        
        # 3. Process with Gemini Natively (Audio In -> Text Out)
        user_text, ai_text_response, current_chat_id = await gemini_service.send_audio_message(
            audio_data=audio_data,
            mime_type=mime_type,
            chat_id=current_chat_id,
            health_context=health_context
        )
        
        logger.info(f"Native Transcription: {user_text}")
        logger.info(f"AI Response: {ai_text_response}")

        # 4. Persist messages to DB
        session = await db.get(ChatSession, current_chat_id)
        if session is None:
            session = ChatSession(id=current_chat_id, user_id=user_id)
            db.add(session)
        elif user_id and not session.user_id:
            session.user_id = user_id
            
        # 5. Convert Response to Audio (TTS) - Save to persistent location
        import os
        from pathlib import Path
        from app.core.config import BASE_DIR
        
        # Create audio directory if it doesn't exist
        audio_dir = BASE_DIR / "media" / "voice_responses"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        audio_filename = f"{current_chat_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.mp3"
        audio_file_path = audio_dir / audio_filename
        audio_output_path = str(audio_file_path)
        
        await audio_service.text_to_speech(ai_text_response, audio_output_path)
        
        # Relative path for database storage
        audio_relative_path = f"voice_responses/{audio_filename}"

        # 6. Persist messages to DB
        # Store audio path in text field as marker since audio_path column doesn't exist
        user_msg = Message(
            chat_session_id=current_chat_id, 
            role="user", 
            text=f"[VOICE_MESSAGE]{user_text}"
        )
        assistant_msg = Message(
            chat_session_id=current_chat_id, 
            role="assistant", 
            text=f"[VOICE_RESPONSE]{ai_text_response}[AUDIO_PATH:{audio_relative_path}]"
        )
        
        db.add_all([user_msg, assistant_msg])
        await db.commit()

        # 7. Encode audio to base64 for immediate response
        with open(audio_output_path, "rb") as audio_file:
            audio_content = audio_file.read()
            audio_base64 = base64.b64encode(audio_content).decode('utf-8')

        return JSONResponse({
            "chat_id": current_chat_id,
            "user_query": user_text,
            "transcript": ai_text_response,
            "audio_base64": audio_base64
        })

    except Exception as e:
        logger.error(f"Voice processing error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Voice processing error: {str(e)}")
    
    finally:
        # No need to clean up - audio files are now stored persistently in media/voice_responses
        pass
