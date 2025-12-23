import os
import io
import warnings
import asyncio
from typing import Tuple, List, Optional
from pathlib import Path

from google import genai
from google.genai import types
from pydub import AudioSegment

from app.core.config import settings

# Suppress pydub RuntimeWarning about ffmpeg if it's not found
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub.utils")

class VoiceService:
    """
    Service for handling voice interactions using Gemini Live API.
    """
    
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment variables")
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        # Using the specific model version mentioned in the sample
        self.model_name = "gemini-2.5-flash-native-audio-preview-12-2025"

    async def convert_to_pcm16_mono_16k(self, file_bytes: bytes) -> bytes:
        """
        Convert arbitrary audio to 16-bit PCM mono 16kHz.
        Prioritizes native wave module for WAV files to avoid ffmpeg dependency.
        """
        import wave
        import audioop

        # Try processing as native WAV first (no ffmpeg needed)
        try:
            with io.BytesIO(file_bytes) as wav_io:
                with wave.open(wav_io, 'rb') as wav:
                    # Get current properties
                    n_channels = wav.getnchannels()
                    sampwidth = wav.getsampwidth()
                    framerate = wav.getframerate()
                    frames = wav.readframes(wav.getnframes())

                    # 1. Convert to Mono if needed
                    if n_channels > 1:
                        frames = audioop.tomono(frames, sampwidth, 1, 0)
                        n_channels = 1

                    # 2. Resample to 16kHz if needed
                    if framerate != 16000:
                        frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 16000, None)
                        framerate = 16000

                    # 3. Convert to 16-bit (2 bytes) if needed
                    if sampwidth != 2:
                        frames = audioop.lin2lin(frames, sampwidth, 2)
                        sampwidth = 2
                    
                    return frames
        except (wave.Error, EOFError):
            # Not a valid WAV file or wave module failed, fall back to pydub
            pass
        except Exception as e:
            print(f"Native WAV conversion failed: {e}, falling back to pydub")
            pass

        # Fallback to pydub (requires ffmpeg for non-WAV or complex conversions)
        try:
            audio = AudioSegment.from_file(io.BytesIO(file_bytes))
            
            # Convert to mono 16kHz 16-bit PCM
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            
            out_buffer = io.BytesIO()
            audio.export(out_buffer, format="s16le")
            return out_buffer.getvalue()
        except Exception as e:
             if "ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower():
                 raise RuntimeError("FFmpeg is missing. Please ensure the audio is sent as a standard WAV file (PCM) from the client.") from e
             raise e

    async def call_gemini_live_with_audio(self, pcm_data: bytes) -> Tuple[bytes, List[str]]:
        """
        Open a Live API session, send PCM audio once, collect full audio response,
        and return it as raw 16-bit PCM at 24kHz.
        """
        # Validate audio data
        if not pcm_data or len(pcm_data) < 1000:  # At least ~30ms of audio at 16kHz
            raise ValueError(f"Audio data too short: {len(pcm_data)} bytes. Please record at least 1 second of audio.")

        # Create proper Content object for system_instruction
        system_instruction_content = types.Content(
            parts=[types.Part(text="You are a helpful and friendly diabetes health assistant. Always respond with audio, never text-only. Keep your responses concise and supportive.")]
        )

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],  # Explicitly request AUDIO only
            system_instruction=system_instruction_content,
        )

        response_audio_chunks: List[bytes] = []
        text_responses: List[str] = []

        async with self.client.aio.live.connect(
            model=self.model_name,
            config=config,
        ) as session:
            try:
                # Create Blob with proper MIME type
                # The SDK expects 'mime_type' and 'data' directly in the content part for real-time input
                # or uses specific methods depending on the SDK version.
                # For google-genai 0.2+, the pattern for audio input in live sessions is using session.send()
                
                # Send audio data using 'send' with a dictionary structure which is safer
                await session.send(
                    input={"role": "user", "parts": [{"inline_data": {"mime_type": "audio/pcm;rate=16000", "data": pcm_data}}]},
                    end_of_turn=True  # Signal that this is the complete user input
                )
                
            except Exception as send_err:
                raise RuntimeError(f"Failed to send audio to Gemini Live API: {str(send_err)}") from send_err

            # Wait for response with timeout
            start_time = asyncio.get_event_loop().time()
            timeout_seconds = 30
            seen_chunks = set()
            
            try:
                async for message in session.receive():
                    # Check for audio data in various locations
                    if hasattr(message, 'server_content') and message.server_content:
                        if hasattr(message.server_content, 'model_turn') and message.server_content.model_turn:
                            if hasattr(message.server_content.model_turn, 'parts'):
                                for part in message.server_content.model_turn.parts:
                                    # Audio in inline_data
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        if hasattr(part.inline_data, 'data') and isinstance(part.inline_data.data, bytes):
                                            chunk_data = part.inline_data.data
                                            chunk_hash = hash(chunk_data[:100])
                                            if chunk_hash not in seen_chunks:
                                                seen_chunks.add(chunk_hash)
                                                response_audio_chunks.append(chunk_data)
                                    
                                    # Text trace (for debugging)
                                    if hasattr(part, 'text') and part.text:
                                        text_responses.append(part.text[:100])

                        # Direct data on server_content
                        if hasattr(message.server_content, 'data') and message.server_content.data:
                            if isinstance(message.server_content.data, bytes):
                                chunk_data = message.server_content.data
                                chunk_hash = hash(chunk_data[:100])
                                if chunk_hash not in seen_chunks:
                                    seen_chunks.add(chunk_hash)
                                    response_audio_chunks.append(chunk_data)

                        # Generation complete signal
                        if hasattr(message.server_content, "generation_complete") and message.server_content.generation_complete:
                            await asyncio.sleep(0.5)  # Wait for any trailing chunks
                            break
                    
                    # Message-level data
                    if hasattr(message, 'data') and message.data is not None:
                        if isinstance(message.data, bytes):
                            chunk_data = message.data
                            chunk_hash = hash(chunk_data[:100])
                            if chunk_hash not in seen_chunks:
                                seen_chunks.add(chunk_hash)
                                response_audio_chunks.append(chunk_data)
                    
                    if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                        break

            except Exception as recv_err:
                # If we have chunks, we can proceed, otherwise re-raise
                if not response_audio_chunks:
                    raise

        if not response_audio_chunks:
            error_msg = "No audio response received from Gemini Live API."
            if text_responses:
                error_msg += f" Text responses received (model might be refusing to speak): {text_responses}"
            raise RuntimeError(error_msg)

        return b"".join(response_audio_chunks), text_responses

    def wrap_pcm24k_to_wav(self, pcm_24k: bytes) -> str:
        """
        Wrap raw 24kHz PCM into a temporary WAV file and return its path.
        """
        from tempfile import NamedTemporaryFile
        import wave

        tmp = NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp.name
        tmp.close()

        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(24000)
            wf.writeframes(pcm_24k)

        return tmp_path
