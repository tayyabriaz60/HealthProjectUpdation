"""
Gemini SDK integration and business logic.
Handles all interactions with Google's Gemini API using the latest SDK with chat sessions.
"""
from google import genai
from app.core.config import settings
from app.core.logging_config import logger
from typing import Dict, Optional, Any
from datetime import datetime
import uuid
import re

try:
    from google.genai import types as genai_types
except Exception:
    genai_types = None

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# --- STRUCTURED OUTPUT SCHEMAS ---

class GlucoseData(BaseModel):
    value: float
    unit: Literal["mg/dL", "mmol/L"]

class FoodData(BaseModel):
    meal_name: str
    calories: Optional[int] = None
    carbs_g: Optional[int] = None

class HealthDataExtraction(BaseModel):
    type: Literal["glucose", "food", "none"]
    data: Optional[dict] = None # Will hold either GlucoseData or FoodData structure

class GlucoseImageAnalysis(BaseModel):
    value: float
    unit: Literal["mg/dL", "mmol/L"]
    analysis: str = Field(..., description="3-4 sentences of empathetic professional analysis")

class FoodImageAnalysis(BaseModel):
    meal_name: str
    calories: Optional[int] = None
    carbs_g: Optional[int] = None
    recommendation_level: Literal["YES", "CAREFUL", "NO"]
    recommendation_text: str = Field(..., description="2-3 lines of plain text advice")

class ImageClassification(BaseModel):
    classification: Literal["GLUCOSE", "FOOD", "OTHER"]

class GeminiService:
    """
    Service class for interacting with Google Gemini API.
    Uses the new SDK's chat API which automatically manages conversation history.
    """
    
    def __init__(self):
        """Initialize Gemini client with API key."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment variables")
        
        # Initialize the new Gemini client
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL_NAME
        self.system_prompt = settings.SYSTEM_PROMPT
        
        # Store active chat sessions in memory with usage tracking
        self.chat_sessions: Dict[str, Any] = {}
        self._session_last_used: Dict[str, datetime] = {}
        self._session_flags: Dict[str, Dict[str, bool]] = {}
    
    def _cleanup_old_sessions(self, max_age_minutes: int = 60, max_sessions: int = 100):
        """
        Remove sessions that are too old or when memory limit is reached.
        Helps maintain server performance.
        """
        now = datetime.utcnow()
        # 1. Remove by age
        to_delete = [
            cid for cid, last_used in self._session_last_used.items()
            if (now - last_used).total_seconds() > max_age_minutes * 60
        ]
        
        # 2. If still too many sessions, remove oldest by last_used
        if len(self.chat_sessions) - len(to_delete) > max_sessions:
            sorted_sessions = sorted(self._session_last_used.items(), key=lambda x: x[1])
            num_more_to_delete = len(self.chat_sessions) - len(to_delete) - max_sessions
            for i in range(num_more_to_delete):
                to_delete.append(sorted_sessions[i][0])

        for cid in set(to_delete):
            self.delete_chat_session(cid)
            if cid in self._session_last_used:
                del self._session_last_used[cid]
            logger.info(f"Session {cid} cleaned up due to age or memory limit.")

    def create_chat_session(self, model_name: Optional[str] = None, history: Optional[list] = None) -> str:
        """
        Create a new chat session with diabetes health assistant system prompt.
        """
        self._cleanup_old_sessions()
        
        model = model_name or self.model_name
        
        # Create chat with system instruction
        system_prompt_applied = False
        try:
            chat = self.client.chats.create(
                model=model,
                system_instruction=self.system_prompt,
                history=history
            )
            system_prompt_applied = True
        except Exception as e:
            print(f"Fallback session creation: {e}")
            chat = self.client.chats.create(model=model, history=history)
            if not history:
                try:
                    chat.send_message(f"Act as a diabetes health assistant: {self.system_prompt}")
                    system_prompt_applied = True
                except: pass
        
        chat_id = str(uuid.uuid4())
        self.chat_sessions[chat_id] = chat
        self._session_last_used[chat_id] = datetime.utcnow()
        self._session_flags[chat_id] = {'system_prompt_applied': system_prompt_applied}
        
        return chat_id

    async def ensure_session(self, chat_id: Optional[str], db: Any) -> str:
        """
        Ensures a session exists. If chat_id is provided but not in memory, 
        it restores it from the database history.
        """
        if not chat_id:
            return self.create_chat_session()

        # If already in memory, update last_used and return
        if chat_id in self.chat_sessions:
            self._session_last_used[chat_id] = datetime.utcnow()
            return chat_id

        # Not in memory -> Restore from DB
        from sqlalchemy import select
        from app.models import Message as MsgModel
        
        stmt = select(MsgModel).where(MsgModel.chat_session_id == chat_id).order_by(MsgModel.created_at.asc())
        res = await db.execute(stmt)
        db_messages = res.scalars().all()

        if db_messages:
            return await self.restore_session_from_db(chat_id, db_messages)
        
        # If no DB history found, create a fresh one
        return self.create_chat_session()

    async def restore_session_from_db(self, chat_id: str, db_messages: list, limit: int = 15) -> str:
        """
        Restores a Gemini chat session object using messages from the database.
        Limit is applied to keep the token usage low and focus on recent context.
        """
        # Always remove existing in-memory session to force a fresh rebuild
        if chat_id in self.chat_sessions:
            del self.chat_sessions[chat_id]
            if chat_id in self._session_flags:
                del self._session_flags[chat_id]
            
        # Convert DB messages to Gemini SDK format, but only the last 'limit' messages
        # We take the last N messages to keep the session concise
        recent_messages = db_messages[-limit:] if len(db_messages) > limit else db_messages

        formatted_history = []
        for msg in recent_messages:
            role = "user" if msg.role == "user" else "model"
            # Skip empty messages or technical markers that shouldn't be in history
            if not msg.text or msg.text == "[Image Uploaded]":
                continue
            formatted_history.append({"role": role, "parts": [{"text": msg.text}]})
            
        # Create a new session with this history
        new_chat_id = self.create_chat_session(history=formatted_history)
        
        # Swap the generated ID with the existing one in memory
        session_obj = self.chat_sessions.pop(new_chat_id)
        self.chat_sessions[chat_id] = session_obj
        
        return chat_id

    def get_chat_session(self, chat_id: str):
        """
        Get an existing chat session.
        
        Args:
            chat_id: Chat session ID
            
        Returns:
            Chat session object
            
        Raises:
            ValueError: If chat session not found
        """
        if chat_id not in self.chat_sessions:
            raise ValueError(f"Chat session {chat_id} not found")
        
        return self.chat_sessions[chat_id]
    
    def _extract_json_from_text(self, text: str) -> str:
        """
        Extracts JSON from text that might be wrapped in markdown code blocks or have extra text.
        Handles cases like: "Here is the JSON:\n```json\n{...}\n```" or "```\n{...}\n```"
        """
        import re
        import json
        
        if not text:
            return ""
        
        # Remove any leading text before JSON (like "Here is the JSON:")
        # Find the first occurrence of { which should be the start of JSON
        first_brace = text.find('{')
        if first_brace > 0:
            text = text[first_brace:]
        
        # Try to find JSON in code blocks first (most common case)
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_block_match:
            candidate = json_block_match.group(1).strip()
            # Validate it's valid JSON
            try:
                json.loads(candidate)
                return candidate
            except:
                pass
        
        # Try to find JSON object directly (starts with { and ends with })
        # Use non-greedy match first, then greedy if that fails
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if not json_match:
            # Fallback to greedy match
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if json_match:
            candidate = json_match.group(0).strip()
            # Validate it's valid JSON
            try:
                json.loads(candidate)
                return candidate
            except:
                # Try to find balanced braces
                brace_count = 0
                start_idx = candidate.find('{')
                if start_idx >= 0:
                    for i in range(start_idx, len(candidate)):
                        if candidate[i] == '{':
                            brace_count += 1
                        elif candidate[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                balanced = candidate[start_idx:i+1]
                                try:
                                    json.loads(balanced)
                                    return balanced
                                except:
                                    pass
                                break
        
        # If no JSON found, return original text (might be plain JSON)
        return text.strip()
    
    @staticmethod
    def _clean_response_text(text: str) -> str:
        """
        Forcefully removes markdown syntax to ensure clean plain text output.
        Removes: bold (**), italics (* or _), headers (#), code blocks (```).
        """
        import re
        if not text:
            return ""
            
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Remove bold/italic markers: **, *, __, _
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) # **text** -> text
        text = re.sub(r'\*(.*?)\*', r'\1', text)     # *text* -> text
        text = re.sub(r'__(.*?)__', r'\1', text)     # __text__ -> text
        text = re.sub(r'_(.*?)_', r'\1', text)       # _text_ -> text
        
        # Remove headers (# Header)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # Collapse multiple newlines/spaces
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    async def send_audio_message(
        self, 
        audio_data: bytes, 
        mime_type: str, 
        chat_id: Optional[str] = None, 
        health_context: Optional[str] = None
    ) -> tuple[str, str, str]:
        """
        Send an audio file to Gemini for native processing (STT + Response).
        Returns a tuple of (transcribed_user_text, ai_response_text, chat_id).
        """
        # Ensure session exists
        if chat_id:
            try:
                chat = self.get_chat_session(chat_id)
            except ValueError:
                chat_id = self.create_chat_session()
                chat = self.chat_sessions[chat_id]
        else:
            chat_id = self.create_chat_session()
            chat = self.chat_sessions[chat_id]

        # 1. Transcribe the audio first (Internal call to get user text for DB logging)
        audio_part = genai_types.Part.from_bytes(data=audio_data, mime_type=mime_type)
        
        # Multilingual transcription - detect any language
        transcribe_prompt = (
            "Transcribe this audio exactly as spoken in the original language. "
            "Preserve the language (Urdu, English, Arabic, etc.). "
            "Return ONLY the transcribed text, nothing else."
        )
        try:
            trans_resp = self.client.models.generate_content(
                model=self.model_name,
                contents=[audio_part, transcribe_prompt]
            )
            user_text = trans_resp.text.strip()
        except Exception as e:
            logger.error(f"Native transcription failed: {e}")
            user_text = "Audio Message" # Fallback

        # 2. Get the actual health assistant response
        # We inject the health context and the audio part
        context_block = f"\n\n[SYSTEM CONTEXT: {health_context}]" if health_context else ""
        
        # Multilingual response prompt - respond in the same language as the user
        multilingual_prompt = (
            f"Please respond to this audio message as a diabetes assistant. "
            f"IMPORTANT: Detect the language spoken in the audio and respond in the SAME language. "
            f"If the user speaks Urdu (or any language using Urdu/Arabic script), respond in Urdu. "
            f"If English, respond in English. Match their language exactly. "
            f"DO NOT respond in Hindi. Always prefer Urdu over Hindi. {context_block}"
        )
        
        # We use the chat session to maintain history
        # Note: Generation config is controlled via system prompt
        response = chat.send_message(
            [audio_part, multilingual_prompt]
        )
        
        clean_ai_text = self._clean_response_text(response.text)
        self._session_last_used[chat_id] = datetime.utcnow()
        
        return user_text, clean_ai_text, chat_id

    async def send_message(self, message: str, chat_id: Optional[str] = None, retry_count: int = 2) -> tuple[str, str]:
        """
        Send a message to the chatbot.
        If chat_id is provided, continues existing conversation.
        If not provided, creates a new chat session.
        
        Args:
            message: User's input message
            chat_id: Optional chat session ID for multi-turn conversations
            retry_count: Number of retries for temporary errors (default: 2)
            
        Returns:
            Tuple of (response_text, chat_id)
        """
        import asyncio
        import time
        
        last_error = None
        
        for attempt in range(retry_count + 1):
            try:
                # Get or create chat session
                if chat_id:
                    try:
                        chat = self.get_chat_session(chat_id)
                    except ValueError:
                        # If client sent stale chat_id, start a fresh session
                        chat_id = self.create_chat_session()
                        chat = self.chat_sessions[chat_id]
                else:
                    chat_id = self.create_chat_session()
                    chat = self.chat_sessions[chat_id]
                
                # Check if this is a follow-up about an image from history
                # If history contains image notes, remind the model
                history = self.get_chat_history(chat_id)
                image_context = ""
                
                # Scan history in reverse to find ONLY the most recent image context
                # We stop as soon as we find the FIRST match from the end (latest one)
                for msg in reversed(history):
                    text_content = msg.get("text", "")
                    
                    # Check for our specific markers
                    # Updated to include [IMAGE_MEMORY]
                    has_marker = (
                        "[IMAGE_MEMORY]" in text_content or
                        "IMAGE_ANALYSIS_CONTEXT" in text_content or 
                        "Detected: Glucose Meter" in text_content or 
                        "Detected: Food" in text_content or
                        "Detected: Unknown" in text_content or
                        "The user uploaded an image" in text_content
                    )
                    
                    if has_marker:
                         image_context = text_content
                         break # Found the latest image! STOP searching.
                
                # Check if the user is asking about an image
                normalized_msg = message.lower()
                is_asking_about_image = any(k in normalized_msg for k in [
                    "image", "photo", "picture", "what is this", "see", "look like", 
                    "summary", "summarize", "detect", "earlier", "showed"
                ])

                if image_context and is_asking_about_image:
                     # FORCEFULLY inject the context again so the model cannot ignore it
                     message = (
                         f"[SYSTEM NOTE: The user is asking about an image they previously uploaded. "
                         f"You cannot see the image file directly, but here is the STORED MEMORY of it that you MUST use to answer.\n"
                         f"Please be PROFESSIONAL, CONCISE, and SIMPLE. Do not list raw data fields.\n"
                         f"CONTEXT:\n{image_context}]\n\n"
                         f"User Question: {message}"
                     )

                # Send message - SDK automatically includes full conversation history
                # System prompt is already applied during session creation
                # Note: Generation config (max_output_tokens, temperature) is controlled via system prompt
                response = chat.send_message(message)
                
                # Clean response (strip markdown)
                clean_text = self._clean_response_text(response.text)
                
                return clean_text, chat_id
                
            except Exception as e:
                error_str = str(e)
                last_error = e
                
                # Check if it's a retryable error (503, 429)
                is_retryable = ('503' in error_str or 'UNAVAILABLE' in error_str or 
                               '429' in error_str or 'RATE_LIMIT' in error_str)
                
                if is_retryable and attempt < retry_count:
                    # Wait before retrying (exponential backoff)
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s...
                    print(f"Retryable error detected. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retry_count + 1})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Handle specific Gemini API errors
                    if '503' in error_str or 'UNAVAILABLE' in error_str:
                        raise Exception("Gemini API is temporarily overloaded. Please try again in a few moments. Tip: Try using streaming mode.")
                    elif '429' in error_str or 'RATE_LIMIT' in error_str:
                        raise Exception("Rate limit exceeded. Please wait a moment before trying again.")
                    elif '401' in error_str or 'UNAUTHENTICATED' in error_str:
                        raise Exception("API key is invalid or expired. Please check your GEMINI_API_KEY.")
                    elif '400' in error_str or 'INVALID_ARGUMENT' in error_str:
                        raise Exception("Invalid request. Please check your message and try again.")
                    else:
                        raise Exception(f"Error generating response: {error_str}")
        
        # If all retries failed
        raise last_error
    
    async def send_message_stream(self, message: str, chat_id: Optional[str] = None):
        """
        Send a message with streaming response.
        
        Args:
            message: User's input message
            chat_id: Optional chat session ID for multi-turn conversations
            
        Yields:
            Response chunks as they are generated
        """
        try:
            # Get or create chat session
            if chat_id:
                try:
                    chat = self.get_chat_session(chat_id)
                except ValueError:
                    chat_id = self.create_chat_session()
                    chat = self.chat_sessions[chat_id]
            else:
                chat_id = self.create_chat_session()
                chat = self.chat_sessions[chat_id]
            
            # Send message with streaming - SDK automatically includes full history
            # System prompt is already applied during session creation
            response_stream = chat.send_message_stream(message)
            
            for chunk in response_stream:
                yield chunk.text, chat_id
                
        except Exception as e:
            error_str = str(e)
            # Handle specific Gemini API errors
            if '503' in error_str or 'UNAVAILABLE' in error_str:
                raise Exception("Gemini API is temporarily overloaded. Please try again in a few moments.")
            elif '429' in error_str or 'RATE_LIMIT' in error_str:
                raise Exception("Rate limit exceeded. Please wait a moment before trying again.")
            elif '401' in error_str or 'UNAUTHENTICATED' in error_str:
                raise Exception("API key is invalid or expired. Please check your GEMINI_API_KEY.")
            elif '400' in error_str or 'INVALID_ARGUMENT' in error_str:
                raise Exception("Invalid request. Please check your message and try again.")
            else:
                raise Exception(f"Error generating streaming response: {error_str}")
    
    def get_chat_history(self, chat_id: str) -> list[dict]:
        """
        Get the conversation history for a chat session.
        Follows official Gemini SDK pattern: chat.get_history()
        
        Official pattern from docs:
        for message in chat.get_history():
            print(f'role - {message.role}', end=": ")
            print(message.parts[0].text)
        
        Args:
            chat_id: Chat session ID
            
        Returns:
            List of messages with role and text
        """
        chat = self.get_chat_session(chat_id)
        
        try:
            # Get history from chat object (official SDK method)
            # SDK automatically maintains full conversation history
            history = []
            
            # Iterate through history messages (official pattern)
            for message in chat.get_history():
                try:
                    # Official pattern: message.role and message.parts[0].text
                    role = getattr(message, 'role', 'unknown')
                    text = ""
                    
                    # Safely extract text from message parts
                    if hasattr(message, 'parts') and message.parts:
                        parts_texts = []
                        for part in message.parts:
                            if hasattr(part, 'text'):
                                parts_texts.append(part.text)
                            elif hasattr(part, 'content'):
                                parts_texts.append(part.content)
                            elif hasattr(part, 'mime_type') and 'image' in part.mime_type:
                                parts_texts.append("[Image Uploaded]")
                            else:
                                parts_texts.append(str(part))
                        text = " ".join(parts_texts)
                        
                    elif hasattr(message, 'text'):
                        text = message.text
                    elif hasattr(message, 'content'):
                        text = message.content
                    
                    history.append({
                        "role": role,
                        "text": str(text) if text else ""
                    })
                except Exception as e:
                    # Skip messages that can't be parsed
                    # Log error for debugging but continue
                    continue
            
            return history
        except AttributeError:
            # If get_history doesn't exist, return empty list
            return []
        except Exception as e:
            # Return empty list on any error
            return []
    
    def delete_chat_session(self, chat_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            chat_id: Chat session ID
            
        Returns:
            True if deleted, False if not found
        """
        if chat_id in self.chat_sessions:
            del self.chat_sessions[chat_id]
            # Clean up session flags
            if hasattr(self, '_session_flags') and chat_id in self._session_flags:
                del self._session_flags[chat_id]
            return True
        return False

    def extract_health_data_from_text(self, text: str) -> dict:
        """
        Analyze user text to extract health data (glucose readings or food logs).
        Uses Gemini JSON mode for reliable extraction.
        """
        prompt = (
            "Analyze the following user message for health data. Return ONLY valid JSON.\n"
            "Rules:\n"
            "1. If the user mentions a glucose/sugar level (e.g., 'my sugar is 120', 'glucose 5.5', 'I checked my glucose and it's 250', 'blood sugar 180'), extract it.\n"
            "2. If no unit is mentioned, assume 'mg/dL' for values > 10, or 'mmol/L' for values <= 10.\n"
            "3. If the user is reporting a meal (e.g., 'I ate a burger', 'had oatmeal'), extract it.\n"
            "4. Return JSON with structure: {\"type\": \"glucose\"|\"food\"|\"none\", \"data\": {...}}\n"
            "5. For glucose: {\"type\": \"glucose\", \"data\": {\"value\": number, \"unit\": \"mg/dL\"|\"mmol/L\"}}\n"
            "6. For food: {\"type\": \"food\", \"data\": {\"meal_name\": string, \"calories\": number|null, \"carbs_g\": number|null}}\n"
            "7. If no health data found: {\"type\": \"none\", \"data\": {}}\n"
            f"User Message: {text}\n"
            "IMPORTANT: Return ONLY the JSON object, no explanatory text."
        )

        try:
            # Try without response_schema first (more reliable)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                }
            )
            
            # Extract JSON manually
            import json
            json_text = self._extract_json_from_text(response.text)
            parsed = json.loads(json_text)
            
            # Ensure the structure matches HealthDataExtraction
            # Handle case where response might have different structure
            if not isinstance(parsed, dict):
                return {"type": "none", "data": {}}
            
            # Normalize the response structure
            result = {
                "type": parsed.get("type", "none"),
                "data": parsed.get("data", {})
            }
            
            # If type is directly in parsed but not in expected format
            if "type" not in result or result["type"] not in ["glucose", "food", "none"]:
                # Try to infer type from data structure
                glucose_value = parsed.get("value") or parsed.get("blood_glucose_value") or parsed.get("glucose_value") or parsed.get("glucose")
                if glucose_value is not None:
                    result["type"] = "glucose"
                    # Auto-detect unit if not provided: > 10 = mg/dL, <= 10 = mmol/L
                    unit = parsed.get("unit", "mg/dL")
                    if not unit or unit not in ["mg/dL", "mmol/L"]:
                        glucose_val = float(glucose_value) if glucose_value else 0
                        unit = "mmol/L" if glucose_val <= 10 else "mg/dL"
                    result["data"] = {
                        "value": float(glucose_value),
                        "unit": unit
                    }
                elif "meal_name" in parsed or parsed.get("type") == "food":
                    result["type"] = "food"
                    result["data"] = {
                        "meal_name": parsed.get("meal_name", "Unidentified Meal"),
                        "calories": parsed.get("calories"),
                        "carbs_g": parsed.get("carbs_g")
                    }
                else:
                    result["type"] = "none"
                    result["data"] = {}
            
            # Validate glucose data structure
            if result["type"] == "glucose" and result.get("data"):
                data = result["data"]
                if not isinstance(data, dict):
                    result["data"] = {}
                    result["type"] = "none"
                elif "value" not in data or not data.get("value"):
                    result["type"] = "none"
                    result["data"] = {}
                else:
                    # Ensure unit is set
                    if "unit" not in data or data["unit"] not in ["mg/dL", "mmol/L"]:
                        glucose_val = float(data["value"])
                        data["unit"] = "mmol/L" if glucose_val <= 10 else "mg/dL"
            
            return result
        except Exception as e:
            print(f"Error extracting health data: {e}")
            import traceback
            print(traceback.format_exc())
            return {"type": "none", "data": {}}

    def analyze_glucose_image(self, image_data: bytes, mime_type: Optional[str] = None) -> dict:
        """
        Analyze a glucose meter image and return parsed value, unit, and brief analysis.
        Uses Gemini JSON mode for structured multimodal output.
        """
        if not image_data:
            raise ValueError("Image file is empty")
        if genai_types is None:
            raise RuntimeError("google.genai.types not available for image handling")

        prompt = (
            "Analyze this glucose meter image and return ONLY valid JSON. "
            "1. Read the blood glucose numerical value and the unit (mg/dL or mmol/L) from this image.\n"
            "2. If the image is a meter, look for the largest numbers on the screen.\n"
            "3. Provide a 2-sentence empathetic analysis based on the reading.\n"
            "Safety: If reading <70 mg/dL or <3.9 mmol/L, emphasize fast-acting carbs immediately.\n"
            "IMPORTANT: Return ONLY the JSON object, no explanatory text before or after."
        )

        image_part = genai_types.Part.from_bytes(
            data=image_data,
            mime_type=mime_type or "image/png"
        )

        import json
        
        # Try without response_schema first (more reliable)
        # response_schema sometimes causes issues where Gemini returns text but no JSON
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"role": "user", "parts": [{"text": prompt}, image_part]}],
                config={
                    'response_mime_type': 'application/json',
                    'max_output_tokens': 500,  # Increased to ensure full JSON response
                }
            )
        except Exception as e:
            print(f"Error generating content: {e}")
            raise ValueError(f"Failed to get response from Gemini: {str(e)}")
        
        # Manual fallback - extract JSON from markdown/text if needed
        try:
            # Check if response has multiple parts or candidates
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Debug: Print full response to understand structure
            print(f"DEBUG: Full response text length: {len(response_text)}")
            print(f"DEBUG: Full response text: {response_text}")
            
            # Check if response was truncated
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'finish_reason'):
                        print(f"DEBUG: Finish reason: {candidate.finish_reason}")
                    if hasattr(candidate, 'safety_ratings'):
                        print(f"DEBUG: Safety ratings: {candidate.safety_ratings}")
            
            # Try to extract JSON
            json_text = self._extract_json_from_text(response_text)
            
            # If extraction failed, check if there's JSON in response parts
            if not json_text or not json_text.strip().startswith('{'):
                # Check if response has parts attribute
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    json_text = self._extract_json_from_text(part.text)
                                    if json_text and json_text.strip().startswith('{'):
                                        break
                        if json_text and json_text.strip().startswith('{'):
                            break
                
                # If still no JSON, try to find it in the raw response string
                if not json_text or not json_text.strip().startswith('{'):
                    # Look for JSON pattern more aggressively
                    import re
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    matches = re.findall(json_pattern, response_text, re.DOTALL)
                    for match in matches:
                        try:
                            test_parsed = json.loads(match)
                            if "value" in test_parsed and "unit" in test_parsed:
                                json_text = match
                                break
                        except:
                            continue
            
            # Validate JSON before parsing
            if not json_text or not json_text.strip().startswith('{'):
                raise ValueError(f"Invalid JSON response from Gemini. Response: {response_text[:500]}")
            
            parsed = json.loads(json_text)
            
            # Handle different field name variations from Gemini
            # Gemini might return: "value", "blood_glucose_value", "glucose_value", etc.
            glucose_value = None
            if "value" in parsed:
                glucose_value = parsed["value"]
            elif "blood_glucose_value" in parsed:
                glucose_value = parsed["blood_glucose_value"]
            elif "glucose_value" in parsed:
                glucose_value = parsed["glucose_value"]
            elif "reading" in parsed:
                glucose_value = parsed["reading"]
            
            # Handle unit field variations
            unit = None
            if "unit" in parsed:
                unit = parsed["unit"]
            elif "blood_glucose_unit" in parsed:
                unit = parsed["blood_glucose_unit"]
            elif "glucose_unit" in parsed:
                unit = parsed["glucose_unit"]
            
            # Validate required fields
            if glucose_value is None:
                raise ValueError(f"Missing glucose value in JSON. Got fields: {list(parsed.keys())}")
            
            # Auto-detect unit if not provided
            if not unit or unit not in ["mg/dL", "mmol/L"]:
                glucose_val = float(glucose_value)
                unit = "mmol/L" if glucose_val <= 10 else "mg/dL"
            else:
                unit = str(unit).strip()
            
            return {
                "value": float(glucose_value),
                "unit": unit,
                "analysis": self._clean_response_text(parsed.get("analysis", "")),
                "raw_response": response_text
            }
        except json.JSONDecodeError as e:
            print(f"JSON decode error in analyze_glucose_image: {e}")
            print(f"Response text: {response.text[:500] if 'response' in locals() and hasattr(response, 'text') else 'N/A'}")
            raise ValueError(f"Failed to parse JSON from Gemini response: {str(e)}")
        except Exception as e:
            print(f"Error in analyze_glucose_image: {e}")
            import traceback
            print(traceback.format_exc())
            raise ValueError(f"Failed to analyze glucose image: {str(e)}")

    def analyze_food_image(self, image_data: bytes, mime_type: Optional[str] = None, health_context: Optional[str] = None) -> dict:
        """
        Analyze a food image and return meal name, estimated calories and recommendation.
        Uses Gemini JSON mode for reliable multimodal analysis.
        """
        if not image_data:
            raise ValueError("Image file is empty")
        if genai_types is None:
            raise RuntimeError("google.genai.types not available for image handling")

        context_part = f"\n\nPatient Health Context:\n{health_context}" if health_context else ""

        prompt = (
            "You are a diabetes nutrition assistant. Analyze this food image and return ONLY valid JSON.\n"
            "Required JSON structure:\n"
            "{\n"
            '  "meal_name": "name of the meal",\n'
            '  "calories": number or null,\n'
            '  "carbs_g": number or null,\n'
            '  "recommendation_level": "YES" or "CAREFUL" or "NO",\n'
            '  "recommendation_text": "2-3 line advice"\n'
            "}\n"
            "Rules:\n"
            "- meal_name: Simple name (e.g., 'Chicken Rice', 'Pasta', 'Salad')\n"
            "- recommendation_level: YES (safe), CAREFUL (moderate), NO (avoid)\n"
            "- Use Patient Health Context for recommendation (if glucose >180, be stricter)\n"
            "- recommendation_text: 2-3 lines max, concise and friendly\n"
            "IMPORTANT: Return ONLY the JSON object, no arrays, no nested objects, no explanatory text."
            f"{context_part}"
        )

        image_part = genai_types.Part.from_bytes(
            data=image_data,
            mime_type=mime_type or "image/png"
        )

        # Try without response_schema first (more reliable)
        # response_schema causes issues with Optional fields
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"role": "user", "parts": [{"text": prompt}, image_part]}],
                config={
                    'response_mime_type': 'application/json',
                    # Removed max_output_tokens to allow full response (default is usually 8192)
                }
            )
        except Exception as e:
            print(f"Error generating content: {e}")
            raise ValueError(f"Failed to get response from Gemini: {str(e)}")
        
        # Extract and parse JSON from response
        try:
            # Get response text - check multiple possible locations
            response_text = ""
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Try to get text from candidates
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                response_text += part.text + "\n"
            
            if not response_text:
                raise ValueError("No text found in Gemini response")
            
            # Check if response was truncated
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                        print(f"DEBUG: Food image finish reason: {finish_reason}")
                        if finish_reason and finish_reason != "STOP":
                            print(f"WARNING: Response may be incomplete. Finish reason: {finish_reason}")
            
            print(f"DEBUG: Food image response text length: {len(response_text)}")
            print(f"DEBUG: Food image response text: {response_text[:1000]}")
            
            # Try to extract JSON
            import json
            json_text = self._extract_json_from_text(response_text)
            
            # If extraction failed, try more aggressive search
            if not json_text or not json_text.strip().startswith('{'):
                # Look for JSON pattern more aggressively in the full response
                import re
                # Try to find any JSON-like structure
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(json_pattern, response_text, re.DOTALL)
                for match in reversed(matches):  # Try longest matches first
                    try:
                        test_parsed = json.loads(match)
                        if "meal_name" in test_parsed or "recommendation_level" in test_parsed:
                            json_text = match
                            break
                    except:
                        continue
            
            # Validate JSON before parsing
            if not json_text or not json_text.strip().startswith('{'):
                # If JSON is incomplete (MAX_TOKENS), try to extract partial data
                if response_text.strip().startswith('{'):
                    # Try to extract partial JSON and complete it
                    import re
                    # Try to extract meal_name even if JSON is incomplete
                    meal_name_match = re.search(r'"meal_name"\s*:\s*"([^"]*)', response_text)
                    meal_name = meal_name_match.group(1) if meal_name_match else None
                    
                    # Try to extract other fields
                    calories_match = re.search(r'"calories"\s*:\s*(\d+)', response_text)
                    calories = int(calories_match.group(1)) if calories_match else None
                    
                    carbs_match = re.search(r'"carbs_g"\s*:\s*(\d+)', response_text)
                    carbs_g = int(carbs_match.group(1)) if carbs_match else None
                    
                    rec_level_match = re.search(r'"recommendation_level"\s*:\s*"([^"]*)', response_text)
                    rec_level = rec_level_match.group(1) if rec_level_match else "CAREFUL"
                    
                    # If we found at least meal_name, return partial data
                    if meal_name:
                        return {
                            "meal_name": meal_name,
                            "calories": calories,
                            "carbs_g": carbs_g,
                            "recommendation_level": rec_level if rec_level in ["YES", "CAREFUL", "NO"] else "CAREFUL",
                            "recommendation_text": "Response was incomplete. Please try again for full analysis.",
                            "raw_response": response_text
                        }
                    
                    # Try to find the last complete JSON object
                    json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                    if json_objects:
                        # Try the longest one
                        for obj in reversed(sorted(json_objects, key=len)):
                            try:
                                test = json.loads(obj)
                                if "meal_name" in test or "recommendation_level" in test:
                                    json_text = obj
                                    break
                            except:
                                continue
                
                if not json_text or not json_text.strip().startswith('{'):
                    raise ValueError(f"Invalid JSON response from Gemini. Response: {response_text[:500]}")
            
            parsed = json.loads(json_text)
            
            # Validate required fields
            if "meal_name" not in parsed and "recommendation_level" not in parsed:
                # Check if it's a different structure (e.g., food_items array)
                if "food_items" in parsed and isinstance(parsed["food_items"], list) and len(parsed["food_items"]) > 0:
                    # Extract from first food item
                    first_item = parsed["food_items"][0]
                    parsed = {
                        "meal_name": first_item.get("name") or first_item.get("meal_name") or "Unidentified Meal",
                        "calories": first_item.get("calories"),
                        "carbs_g": first_item.get("carbs_g") or first_item.get("carbs"),
                        "recommendation_level": parsed.get("recommendation_level", "CAREFUL"),
                        "recommendation_text": parsed.get("recommendation_text") or first_item.get("recommendation", "")
                    }
                else:
                    raise ValueError(f"Missing required fields in JSON: {parsed}")
            
            return {
                "meal_name": parsed.get("meal_name", "Unidentified Meal"),
                "calories": parsed.get("calories"),  # Can be None
                "carbs_g": parsed.get("carbs_g"),  # Can be None
                "recommendation_level": parsed.get("recommendation_level", "CAREFUL"),
                "recommendation_text": self._clean_response_text(parsed.get("recommendation_text", "")),
                "raw_response": response_text
            }
        except json.JSONDecodeError as e:
            print(f"JSON decode error in analyze_food_image: {e}")
            print(f"Response text: {response_text[:1000] if 'response_text' in locals() else 'N/A'}")
            return {
                "meal_name": "Unidentified Meal",
                "recommendation_level": "CAREFUL",
                "recommendation_text": "Could not analyze the image.",
                "raw_response": str(e)
            }
        except Exception as e:
            print(f"Error in analyze_food_image: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                "meal_name": "Unidentified Meal",
                "recommendation_level": "CAREFUL",
                "recommendation_text": "Could not analyze the image.",
                "raw_response": str(e)
            }

    def analyze_general_image(self, image_data: bytes, mime_type: Optional[str] = None) -> dict:
        """
        Analyze a general image that is neither glucose meter nor food.
        """
        if not image_data:
            raise ValueError("Image file is empty")
        if genai_types is None:
            raise RuntimeError("google.genai.types not available for image handling")

        prompt = (
            "Analyze this image and provide a brief, helpful description. "
            "If it relates to health, fitness, or lifestyle, mention that connection. "
            "Keep the response concise (2-3 sentences)."
        )

        image_part = genai_types.Part.from_bytes(
            data=image_data,
            mime_type=mime_type or "image/png"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[{"role": "user", "parts": [{"text": prompt}, image_part]}]
        )

        description = getattr(response, "text", "") or "I see an image but I'm not sure what it is."

        return {
            "description": description,
            "raw_response": description
        }

    def analyze_image_auto(
        self,
        image_data: bytes,
        mime_type: Optional[str] = None,
        health_context: Optional[str] = None
    ) -> dict:
        """
        Auto-detect whether the image is a glucose meter, food, or something else.
        Uses Gemini JSON mode for reliable classification.
        """
        if not image_data:
            raise ValueError("Image file is empty")
        if genai_types is None:
            raise RuntimeError("google.genai.types not available for image handling")

        image_part = genai_types.Part.from_bytes(
            data=image_data,
            mime_type=mime_type or "image/png"
        )

        # Step 1: classify the image (glucose meter vs food vs other)
        classify_prompt = (
            "Determine the category of this image for a diabetes health app.\n"
            "- GLUCOSE: A blood glucose meter device (Glucometer), a continuous glucose monitor (CGM) sensor on skin, "
            "or a mobile app screen showing blood sugar numbers (e.g., 120 mg/dL, 6.5 mmol/L).\n"
            "- FOOD: Any edible meal, individual food item, beverage (except plain water), "
            "or a food logging app screenshot showing a meal with calories.\n"
            "- OTHER: Any image that is NOT a medical device, blood sugar reading, or food (e.g., people, pets, "
            "landscapes, documents, cars, or generic household objects).\n\n"
            "If the image is blurry or unidentifiable, choose OTHER."
        )

        try:
            classify_resp = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"role": "user", "parts": [{"text": classify_prompt}, image_part]}],
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': ImageClassification,
                }
            )
            
            classification = "OTHER"
            if hasattr(classify_resp, 'parsed') and classify_resp.parsed:
                classification = classify_resp.parsed.classification
            else:
                import json
                classification = json.loads(classify_resp.text).get("classification", "OTHER")

        except Exception as e:
            print(f"Classification error: {e}")
            classification = "OTHER"

        # Branch based on classification
        if classification == "GLUCOSE":
            reading = self.analyze_glucose_image(
                image_data=image_data,
                mime_type=mime_type
            )
            return {
                "type": "glucose",
                "reading": {"value": reading["value"], "unit": reading["unit"]},
                "analysis": reading.get("analysis"),
                "raw_response": reading.get("raw_response"),
                "memory_summary": (
                    f"CATEGORY: Glucose Meter\n"
                    f"SUMMARY: An image of a digital glucose meter.\n"
                    f"DETAILS: Reading: {reading['value']} {reading['unit']}. Analysis: {reading.get('analysis')}"
                )
            }

        elif classification == "FOOD":
            meal = self.analyze_food_image(
                image_data=image_data,
                mime_type=mime_type,
                health_context=health_context
            )
            return {
                "type": "food",
                "meal": {
                    "meal_name": meal.get("meal_name"),
                    "calories": meal.get("calories"),
                    "carbs_g": meal.get("carbs_g")
                },
                "recommendation_level": meal.get("recommendation_level"),
                "recommendation": meal.get("recommendation_text"),
                "raw_response": meal.get("raw_response"),
                "memory_summary": (
                    f"CATEGORY: Food\n"
                    f"SUMMARY: An image of a meal identified as {meal.get('meal_name')}.\n"
                    f"DETAILS: Est. Calories: {meal.get('calories')}, Carbs: {meal.get('carbs_g')}g. Recommendation: {meal.get('recommendation_text')}"
                )
            }
            
        else:
            # For OTHER images, still generate a description so AI can answer "what was in image"
            try:
                description_prompt = (
                    "Describe what you see in this image in 1-2 sentences. "
                    "Be specific about what objects, people, or scenes are visible. "
                    "This is NOT a glucose meter or food image."
                )
                desc_resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[{"role": "user", "parts": [{"text": description_prompt}, image_part]}]
                )
                description = desc_resp.text.strip() if desc_resp.text else "An image that is not a glucose meter or food."
            except Exception as e:
                print(f"Error generating description for unknown image: {e}")
                description = "An image that is not a glucose meter or food."
            
            return {
                "type": "unknown",
                "description": description,
                "message": (
                    f"This image shows: {description}. "
                    "This is not a glucose meter or food image. "
                    "Please upload a photo of your meter or your meal so I can assist you with diabetes management."
                ),
                "memory_summary": (
                    f"CATEGORY: Other/Unknown Image\n"
                    f"SUMMARY: User uploaded an image that is not a glucose meter or food.\n"
                    f"DETAILS: {description}"
                )
            }
