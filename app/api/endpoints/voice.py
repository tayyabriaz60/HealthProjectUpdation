from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os

from app.services.voice_service import VoiceService

# Router for voice endpoints
api_router = APIRouter(prefix="/api/voice", tags=["voice"])

# Lazy initialization
_voice_service_instance = None

def get_voice_service() -> VoiceService:
    global _voice_service_instance
    if _voice_service_instance is None:
        _voice_service_instance = VoiceService()
    return _voice_service_instance

@api_router.post("/chat")
async def voice_chat(file: UploadFile = File(...)):
    """
    Accepts an audio file (e.g., from Flutter or browser), sends it to Gemini Live,
    and returns a WAV audio response.
    
    The input audio is converted to the format Gemini expects (16kHz PCM),
    and the response is converted back to a standard WAV file (24kHz).
    """
    try:
        voice_service = get_voice_service()
        file_bytes = await file.read()
        
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file.")

        # Check if it's already PCM (e.g. from browser conversion) or needs conversion
        pcm_16k = None
        if file.content_type == "audio/pcm" or (file.filename and file.filename.endswith(".pcm")):
            pcm_16k = file_bytes
            # Validate PCM format (even length)
            if len(pcm_16k) % 2 != 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid PCM format: data length ({len(pcm_16k)} bytes) is not even."
                )
        else:
            # Convert generic audio (mp3, wav, webm, etc.) to PCM
            try:
                pcm_16k = await voice_service.convert_to_pcm16_mono_16k(file_bytes)
            except Exception as conv_err:
                raise HTTPException(
                    status_code=500,
                    detail=f"Audio conversion failed: {str(conv_err)}"
                ) from conv_err

        # Call Gemini Live
        try:
            pcm_24k, text_responses = await voice_service.call_gemini_live_with_audio(pcm_16k)
        except RuntimeError as gemini_err:
            raise HTTPException(status_code=500, detail=str(gemini_err)) from gemini_err

        # Wrap raw PCM in WAV container for easy playback
        wav_path = voice_service.wrap_pcm24k_to_wav(pcm_24k)
        
        return FileResponse(
            wav_path,
            media_type="audio/wav",
            filename="response.wav",
            # Clean up temp file after sending is not natively supported by FileResponse in older FastAPI versions
            # but modern OS cleans /tmp eventually. For production, a BackgroundTask to delete is better.
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice processing error: {str(e)}")
