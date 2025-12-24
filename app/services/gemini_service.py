"""
Gemini SDK integration and business logic.
Handles all interactions with Google's Gemini API using the latest SDK with chat sessions.
"""
from google import genai
from app.core.config import settings
from typing import Dict, Optional, Any
import uuid
import re

try:
    from google.genai import types as genai_types
except Exception:
    genai_types = None


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
        
        # Store active chat sessions in memory
        # In production, consider using Redis or a database
        self.chat_sessions: Dict[str, Any] = {}
        self._session_flags: Dict[str, Dict[str, bool]] = {}
    
    def create_chat_session(self, model_name: Optional[str] = None) -> str:
        """
        Create a new chat session with diabetes health assistant system prompt.
        
        Args:
            model_name: Optional model name (uses default if not provided)
            
        Returns:
            Unique chat session ID
        """
        model = model_name or self.model_name
        
        # Create chat with system instruction for diabetes health assistant
        # The system instruction guides the AI's behavior throughout the conversation
        system_prompt_applied = False
        try:
            # Try to use system_instruction parameter if available in SDK
            chat = self.client.chats.create(
                model=model,
                system_instruction=self.system_prompt
            )
            system_prompt_applied = True
        except (TypeError, AttributeError, Exception):
            # Fallback: Create chat without system_instruction
            # We'll send system prompt as first message to establish context
            chat = self.client.chats.create(model=model)
            # Send system instruction as first message to establish the assistant's role
            try:
                chat.send_message(f"Please act as a diabetes health assistant. Follow these guidelines:\n\n{self.system_prompt}")
                system_prompt_applied = True
            except Exception:
                # If sending fails, continue without it
                pass
        
        # Generate unique session ID
        chat_id = str(uuid.uuid4())
        
        # Store the chat session and track if system prompt was applied
        self.chat_sessions[chat_id] = chat
        # Store a flag to know if system prompt was applied
        if not hasattr(self, '_session_flags'):
            self._session_flags = {}
        self._session_flags[chat_id] = {'system_prompt_applied': system_prompt_applied}
        
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
                
                # Send message - SDK automatically includes full conversation history
                # System prompt is already applied during session creation
                response = chat.send_message(message)
                
                return response.text, chat_id
                
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
                        if len(message.parts) > 0:
                            part = message.parts[0]
                            if hasattr(part, 'text'):
                                text = part.text
                            elif hasattr(part, 'content'):
                                text = part.content
                            else:
                                text = str(part)
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

    def analyze_glucose_image(self, image_data: bytes, mime_type: Optional[str] = None) -> dict:
        """
        Analyze a glucose meter image and return parsed value, unit, and brief analysis.
        Uses the same Gemini client (multimodal) as the text chat.
        """
        if not image_data:
            raise ValueError("Image file is empty")
        if genai_types is None:
            raise RuntimeError("google.genai.types not available for image handling")

        # Build multimodal prompt with image
        prompt = (
            "Read the blood glucose value from this glucose meter image.\n"
            'Respond ONLY with the number and unit in this exact format: "VALUE UNIT"\n'
            'Examples: "125 mg/dL" or "6.9 mmol/L"\n'
            'If you cannot read it clearly, respond with "Unable to read"'
        )

        image_part = genai_types.Part.from_bytes(
            data=image_data,
            mime_type=mime_type or "image/png"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[{"role": "user", "parts": [{"text": prompt}, image_part]}]
        )

        reading_text = getattr(response, "text", "") or ""
        if "unable" in reading_text.lower() or "cannot" in reading_text.lower():
            raise ValueError("Unable to read glucose meter from image")

        match = re.search(r"(\d+\.?\d*)\s*(mg/dL|mmol/L)", reading_text, re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not parse glucose value from: {reading_text}")

        value = float(match.group(1))
        unit = match.group(2)

        analysis_prompt = (
            f"The patient has a glucose reading of {value} {unit}.\n"
            "Provide a brief health analysis (3-4 sentences):\n"
            "1. Is this reading normal, high, or low?\n"
            "2. What should the patient do next?\n"
            "3. Any immediate concerns?\n"
            "Be empathetic and professional."
        )

        analysis_resp = self.client.models.generate_content(
            model=self.model_name,
            contents=[analysis_prompt]
        )
        analysis_text = getattr(analysis_resp, "text", "") or ""

        return {
            "value": value,
            "unit": unit,
            "analysis": analysis_text,
            "raw_response": reading_text
        }

    def analyze_food_image(self, image_data: bytes, mime_type: Optional[str] = None, health_context: Optional[str] = None) -> dict:
        """
        Analyze a food image and return meal name, estimated calories and recommendation.
        
        Args:
            image_data: Image file bytes
            mime_type: MIME type of the image (optional)
            health_context: Optional health context (e.g., latest glucose reading)
            
        Returns:
            Dictionary with meal_name, calories, recommendation, and raw_response
        """
        if not image_data:
            raise ValueError("Image file is empty")
        if genai_types is None:
            raise RuntimeError("google.genai.types not available for image handling")

        # Build context part if health context is provided
        context_part = f"\n\nPatient Health Context:\n{health_context}" if health_context else ""

        # Build multimodal prompt with image
        prompt = (
            "You are a diabetes nutrition assistant. Analyze this food image and reply ONLY with JSON.\n"
            "Return this JSON shape (no markdown, no extra text):\n"
            "{\n"
            '  "meal_name": "<short name>",\n'
            '  "calories": <number or null>,\n'
            '  "recommendation_level": "YES" | "CAREFUL" | "NO",\n'
            '  "recommendation_text": "<1-2 short sentences, concise, patient-friendly>",\n'
            '  "carbs_g": <number or null>\n'
            "}\n"
            "Rules:\n"
            "- Keep it brief and readable for a patient.\n"
            "- If unsure, set calories or carbs_g to null.\n"
            "- recommendation_level must be exactly YES, CAREFUL, or NO.\n"
            "- Do not include any extra fields or explanations."
            f"{context_part}"
        )

        image_part = genai_types.Part.from_bytes(
            data=image_data,
            mime_type=mime_type or "image/png"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[{"role": "user", "parts": [{"text": prompt}, image_part]}]
        )

        response_text = getattr(response, "text", "") or ""

        meal_name = None
        calories = None
        recommendation_level = None
        recommendation_text = None
        carbs_g = None

        # Try JSON parsing first
        try:
            import json as _json
            parsed = _json.loads(response_text)
            meal_name = parsed.get("meal_name")
            calories = parsed.get("calories")
            recommendation_level = parsed.get("recommendation_level")
            recommendation_text = parsed.get("recommendation_text")
            carbs_g = parsed.get("carbs_g")
        except Exception:
            pass

        # Fallback regex parsing for robustness
        if not meal_name:
            meal_match = re.search(r"meal_name[:=]\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE)
            if meal_match:
                meal_name = meal_match.group(1).strip()
        if not calories:
            calories_match = re.search(r"calories[:=]\s*(\d+)", response_text, re.IGNORECASE)
            if calories_match:
                calories = int(calories_match.group(1))
        if not recommendation_level:
            rec_level_match = re.search(r"(YES|CAREFUL|NO)", response_text, re.IGNORECASE)
            if rec_level_match:
                recommendation_level = rec_level_match.group(1).upper()
        if not recommendation_text:
            rec_text_match = re.search(r"recommendation[_:\-]\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
            if rec_text_match:
                recommendation_text = rec_text_match.group(1).strip()

        # Fallback defaults
        meal_name = meal_name or "Unidentified Meal"
        recommendation_level = (recommendation_level or "CAREFUL").upper()
        recommendation_text = recommendation_text or "Recommendation not available."

        return {
            "meal_name": meal_name,
            "calories": calories,
            "recommendation_level": recommendation_level,
            "recommendation_text": recommendation_text,
            "carbs_g": carbs_g,
            "raw_response": response_text
        }

    def analyze_image_auto(
        self,
        image_data: bytes,
        mime_type: Optional[str] = None,
        health_context: Optional[str] = None
    ) -> dict:
        """
        Auto-detect whether the image is a glucose meter or food, then analyze accordingly.

        Args:
            image_data: Image file bytes
            mime_type: MIME type of the image (optional)
            health_context: Optional health context for food analysis

        Returns:
            Dict containing type ("glucose" or "food") and corresponding analysis payload.
        """
        if not image_data:
            raise ValueError("Image file is empty")
        if genai_types is None:
            raise RuntimeError("google.genai.types not available for image handling")

        image_part = genai_types.Part.from_bytes(
            data=image_data,
            mime_type=mime_type or "image/png"
        )

        # Step 1: classify the image (glucose meter vs food)
        classify_prompt = (
            "Classify this image as exactly one of: GLUCOSE or FOOD.\n"
            "- If it is a glucose meter display with a numeric reading, answer: GLUCOSE\n"
            "- If it is a food/meal, answer: FOOD\n"
            "- If it is ANYTHING ELSE (e.g. selfie, landscape, computer screen, code, blurry, unclear), answer: UNKNOWN\n"
            "Reply with a single word only."
        )

        classify_resp = self.client.models.generate_content(
            model=self.model_name,
            contents=[{"role": "user", "parts": [{"text": classify_prompt}, image_part]}]
        )
        classify_text = (getattr(classify_resp, "text", "") or "").strip().upper()

        classification = "unknown"
        if "GLUCOSE" in classify_text:
            classification = "glucose"
        elif "FOOD" in classify_text and "UNKNOWN" not in classify_text:
            classification = "food"
        
        if classification == "unknown":
            return {
                "success": False,
                "type": "unknown", 
                "message": "Please upload a glucose meter or food image only."
            }

        if classification == "glucose":
            reading = self.analyze_glucose_image(
                image_data=image_data,
                mime_type=mime_type
            )
            return {
                "type": "glucose",
                "reading": {"value": reading["value"], "unit": reading["unit"]},
                "analysis": reading.get("analysis"),
                "raw_response": reading.get("raw_response")
            }

        if classification == "food":
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
                "raw_response": meal.get("raw_response")
            }

        # Instead of error, return a friendly object for frontend to display
        return {
            "type": "unknown",
            "message": "I'm not sure what this is. I can help analyze your glucose meter readings or food photos. Could you try uploading one of those?"
        }
