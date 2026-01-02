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
    
    def create_chat_session(self, model_name: Optional[str] = None, history: Optional[list] = None) -> str:
        """
        Create a new chat session with diabetes health assistant system prompt.
        
        Args:
            model_name: Optional model name (uses default if not provided)
            history: Optional initial history to load into the session
            
        Returns:
            Unique chat session ID
        """
        model = model_name or self.model_name
        
        # Create chat with system instruction for diabetes health assistant
        system_prompt_applied = False
        try:
            # Try to use system_instruction parameter if available in SDK
            chat = self.client.chats.create(
                model=model,
                system_instruction=self.system_prompt,
                history=history # Load history if provided
            )
            system_prompt_applied = True
        except (TypeError, AttributeError, Exception) as e:
            print(f"Fallback session creation due to: {e}")
            # Fallback: Create chat without system_instruction
            chat = self.client.chats.create(model=model, history=history)
            if not history:
                # Send system instruction as first message only if no history exists
                try:
                    chat.send_message(f"Please act as a diabetes health assistant. Follow these guidelines:\n\n{self.system_prompt}")
                    system_prompt_applied = True
                except Exception:
                    pass
        
        # Generate unique session ID
        chat_id = str(uuid.uuid4())
        
        # Store the chat session
        self.chat_sessions[chat_id] = chat
        # Store a flag to know if system prompt was applied
        if not hasattr(self, '_session_flags'):
            self._session_flags = {}
        self._session_flags[chat_id] = {'system_prompt_applied': system_prompt_applied}
        
        return chat_id

    async def restore_session_from_db(self, chat_id: str, db_messages: list) -> str:
        """
        Restores a Gemini chat session object using messages from the database.
        Returns the chat_id.
        """
        # CRITICAL FIX: Always remove existing in-memory session to force a rebuild from DB.
        # This ensures that any new messages added to DB (like image context) are picked up.
        if chat_id in self.chat_sessions:
            del self.chat_sessions[chat_id]
            if chat_id in self._session_flags:
                del self._session_flags[chat_id]
            
        # Convert DB messages to Gemini SDK format
        formatted_history = []
        print(f"DEBUG: Restoring session {chat_id} with {len(db_messages)} messages from DB")
        for msg in db_messages:
            role = "user" if msg.role == "user" else "model"
            # Explicitly log content to verify image context is being loaded
            if "IMAGE_ANALYSIS_CONTEXT" in msg.text:
                print(f"DEBUG: Found IMAGE_ANALYSIS_CONTEXT in DB message: {msg.text[:50]}...")
            
            formatted_history.append({"role": role, "parts": [{"text": msg.text}]})
            
        # Create a new session with this history
        # Note: We keep the same chat_id to maintain consistency
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
        Returns a dict with type ("glucose", "food", or "none") and the extracted data.
        """
        prompt = (
            "Analyze the following user message for health data. "
            "If the user is reporting a glucose level (e.g., 'my sugar is 120', 'glucose 5.5'), extract it.\n"
            "If the user is reporting a meal (e.g., 'I ate a burger', 'had oatmeal for breakfast'), extract it.\n"
            "Return ONLY a JSON object with this structure (no markdown):\n"
            "{\n"
            '  "type": "glucose" | "food" | "none",\n'
            '  "data": {\n'
            '    // If glucose:\n'
            '    "value": <number>,\n'
            '    "unit": "mg/dL" | "mmol/L",\n'
            '    // If food:\n'
            '    "meal_name": "<name>",\n'
            '    "calories": <estimated_number>,\n'
            '    "carbs_g": <estimated_number>\n'
            '  }\n'
            "}\n"
            f"User Message: {text}"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            response_text = getattr(response, "text", "") or "{}"
            
            # Clean up potential markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            import json
            return json.loads(response_text)
        except Exception as e:
            print(f"Error extracting health data: {e}")
            return {"type": "none", "data": {}}

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
            "Read the blood glucose value from this image (it could be a photo of a meter or a screenshot of an app).\n"
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
        analysis_text = self._clean_response_text(analysis_text) # Force clean

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
            "You are a diabetes nutrition assistant. Analyze this food image or screenshot and reply ONLY with JSON.\n"
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
            "- Do not include any extra fields or explanations.\n"
            "- CRITICAL: Use the Patient Health Context (e.g. latest glucose) to determine the recommendation.\n"
            "- If glucose is HIGH (>180), be stricter with carbs/sugar.\n"
            "- If glucose is LOW (<70), suggest fast-acting carbs if appropriate.\n"
            "- STRICTLY NO MARKDOWN in recommendation_text. No bold (**), no italics (*).\n"
            "- Keep recommendation_text strictly between 2 to 3 lines."
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
        Strictly rejects non-medical images.
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
            "Classify this image as exactly one of: GLUCOSE, FOOD, or OTHER.\n"
            "- If it is a glucose meter display, or a screenshot of a medical app showing a glucose reading, answer: GLUCOSE\n"
            "- If it is a food/meal, or a screenshot of a food logging app showing a meal, answer: FOOD\n"
            "- If it is anything else, answer: OTHER\n"
            "Reply with a single word only."
        )

        classify_resp = self.client.models.generate_content(
            model=self.model_name,
            contents=[{"role": "user", "parts": [{"text": classify_prompt}, image_part]}]
        )
        classify_text = (getattr(classify_resp, "text", "") or "").strip().upper()

        classification = "other"
        if "GLUCOSE" in classify_text:
            classification = "glucose"
        elif "FOOD" in classify_text:
            classification = "food"
        
        # Branch based on classification
        if classification == "glucose":
            reading = self.analyze_glucose_image(
                image_data=image_data,
                mime_type=mime_type
            )
            return {
                "type": "glucose",
                "reading": {"value": reading["value"], "unit": reading["unit"]},
                "analysis": reading.get("analysis"),
                "raw_response": reading.get("raw_response"),
                # Generic Memory Summary
                "memory_summary": (
                    f"CATEGORY: Glucose Meter\n"
                    f"SUMMARY: An image of a digital glucose meter.\n"
                    f"DETAILS: Reading: {reading['value']} {reading['unit']}. Analysis: {reading.get('analysis')}"
                )
            }

        elif classification == "food":
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
                # Generic Memory Summary
                "memory_summary": (
                    f"CATEGORY: Food\n"
                    f"SUMMARY: An image of a meal identified as {meal.get('meal_name')}.\n"
                    f"DETAILS: Est. Calories: {meal.get('calories')}, Carbs: {meal.get('carbs_g')}g. Recommendation: {meal.get('recommendation_text')}"
                )
            }
            
        else:
            # STRICT REJECTION for non-medical images as requested
            return {
                "type": "unknown", 
                "message": "Could not analyze this image. Please upload a food image or glucose meter."
            }
