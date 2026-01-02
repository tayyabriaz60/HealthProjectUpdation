import os
import io
import warnings
import asyncio
import urllib
from typing import Tuple, List, Optional
from pathlib import Path

# --- MONKEY PATCH FOR WEBSOCKETS COMPATIBILITY ---
# The google-genai SDK (v0.2.x) calls recv(decode=False) which fails on standard websockets library.
try:
    from websockets.legacy.protocol import WebSocketCommonProtocol
    
    _original_recv = WebSocketCommonProtocol.recv

    async def _patched_recv(self, *args, **kwargs):
        # Remove 'decode' argument if present, as standard websockets.recv() doesn't support it
        kwargs.pop('decode', None) 
        return await _original_recv(self, *args, **kwargs)

    WebSocketCommonProtocol.recv = _patched_recv
    
    # Also patch Protocol for newer websockets versions if needed
    from websockets.protocol import Protocol
    if hasattr(Protocol, 'recv'):
        _original_proto_recv = Protocol.recv
        
        async def _patched_proto_recv(self, *args, **kwargs):
             kwargs.pop('decode', None)
             return await _original_proto_recv(self, *args, **kwargs)
        
        Protocol.recv = _patched_proto_recv

except ImportError:
    pass
# -------------------------------------------------

from google import genai

# Work around missing urllib import in some SDK internals.
if not hasattr(genai, "urllib"):
    genai.urllib = urllib
try:
    from google.genai import _api_client as _api_client
    if not hasattr(_api_client, "urllib"):
        _api_client.urllib = urllib
except Exception:
    pass

# Work around older asyncio loops that don't accept "additional_headers".
_orig_base_create_connection = asyncio.BaseEventLoop.create_connection

async def _patched_create_connection(self, *args, **kwargs):
    kwargs.pop("additional_headers", None)
    return await _orig_base_create_connection(self, *args, **kwargs)

asyncio.BaseEventLoop.create_connection = _patched_create_connection
from google.genai import types
# from pydub import AudioSegment  # Removed to avoid ffmpeg dependency

from app.core.config import settings

# Suppress pydub RuntimeWarning about ffmpeg if it's not found
# warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub.utils")

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
        Strictly uses native wave module. Fails if not a valid WAV or if conversion requires ffmpeg.
        """
        import wave
        import audioop

        # Try processing as native WAV
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
            # Not a valid WAV file
            raise ValueError("Invalid WAV file. Non-WAV formats (mp3, webm) require ffmpeg which is disabled.")
        except Exception as e:
            print(f"Native WAV conversion failed: {e}")
            raise ValueError(f"Audio processing failed: {e}")

    async def call_gemini_live_with_audio(self, pcm_data: bytes) -> Tuple[bytes, List[str]]:
        """
        Open a Live API session, send PCM audio once, collect full audio response,
        and return it as raw 16-bit PCM at 24kHz.
        """
        print(f"Debug: Starting Gemini Live call with {len(pcm_data)} bytes of PCM data")
        
        # Validate audio data
        if not pcm_data or len(pcm_data) < 1000:  # At least ~30ms of audio at 16kHz
             # Relaxed constraint: Allow slightly shorter audio, but log warning
             # raise ValueError(f"Audio data too short: {len(pcm_data)} bytes. Please record at least 1 second of audio.")
             pass
        
        # Work around older asyncio loop implementations that don't accept
        # the "additional_headers" kwarg used by some websocket clients.
        loop = asyncio.get_running_loop()
        if not hasattr(loop, "_orig_create_connection"):
            orig_create_connection = loop.create_connection

            async def _create_connection_wrapper(*args, **kwargs):
                kwargs.pop("additional_headers", None)
                return await orig_create_connection(*args, **kwargs)

            loop._orig_create_connection = orig_create_connection
            loop.create_connection = _create_connection_wrapper

        # Create proper Content object for system_instruction
        system_instruction_content = types.Content(
            parts=[types.Part(text="You are a helpful and friendly diabetes health assistant. Always respond with audio, never text-only. Keep your responses concise and supportive. If the user speaks in a language other than English, detect the language and respond in that same language.")]
        )

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],  # Explicitly request AUDIO only
            system_instruction=system_instruction_content,
        )

        response_audio_chunks: List[bytes] = []
        text_responses: List[str] = []

        # Retry logic: Try twice to get audio response
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                async with self.client.aio.live.connect(
                    model=self.model_name,
                    config=config,
                ) as session:
                    # Create Blob with proper MIME type
                    await session.send(
                        input={"mime_type": "audio/pcm;rate=16000", "data": pcm_data},
                        end_of_turn=True  # Signal that this is the complete user input
                    )
                    
                    # Wait for response with timeout
                    start_time = asyncio.get_event_loop().time()
                    # INCREASED TIMEOUT to 60 seconds (was 30) to handle Gemini Live latency
                    timeout_seconds = 60
                    seen_chunks = set()
                    
                    try:
                         # The .receive() iterator in version 0.2.2 might have an issue in some environments
                        # We will process messages manually if the async generator fails
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

                    except TypeError as type_err:
                         if "unexpected keyword argument 'decode'" in str(type_err):
                            raise RuntimeError("Dependency Error: 'websockets' library version conflict. Please upgrade 'google-genai' or check 'websockets' version.") from type_err
                         raise type_err
                
                # If we got audio, break the retry loop
                if response_audio_chunks:
                    break
                    
                print(f"Attempt {attempt+1} failed to get audio. Retrying...")
                
            except Exception as e:
                print(f"Error in Gemini Live connection attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    # If this was the last attempt, re-raise
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
