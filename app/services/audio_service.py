import edge_tts
import asyncio
import os
import uuid
import re
from typing import Optional

class AudioService:
    """
    Service for handling audio synthesis (TTS).
    Transcription is now handled natively by Gemini 2.5 Flash.
    Supports multiple languages with automatic language detection.
    """
    
    def __init__(self):
        # Default TTS Voice Configuration
        self.default_voice = "en-US-AriaNeural"
        
        # Multilingual voice mapping
        self.voice_map = {
            'ur': 'ur-PK-AsadNeural',      # Urdu (Pakistan)
            'ar': 'ar-SA-ZariyahNeural',   # Arabic (Saudi Arabia)
            'es': 'es-ES-ElviraNeural',    # Spanish (Spain)
            'fr': 'fr-FR-DeniseNeural',    # French (France)
            'de': 'de-DE-KatjaNeural',     # German (Germany)
            'it': 'it-IT-ElsaNeural',      # Italian (Italy)
            'pt': 'pt-BR-FranciscaNeural', # Portuguese (Brazil)
            'zh': 'zh-CN-XiaoxiaoNeural',  # Chinese (Simplified)
            'ja': 'ja-JP-NanamiNeural',     # Japanese (Japan)
            'ko': 'ko-KR-SunHiNeural',     # Korean (Korea)
            'en': 'en-US-AriaNeural',      # English (US) - default
        }

    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns.
        Returns language code (ur, hi, ar, en, etc.)
        """
        # Remove whitespace and get sample
        sample = text.strip()[:200] if len(text) > 200 else text.strip()
        
        if not sample:
            return 'en'
        
        # Urdu/Arabic script detection (Urdu uses Arabic script with Persian characters)
        urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        if urdu_pattern.search(sample):
            # Urdu has more Persian characters, Arabic has more Arabic-specific characters
            persian_chars = re.compile(r'[\u0670-\u06D3\u06D5-\u06FF]')
            if persian_chars.search(sample):
                return 'ur'  # Urdu (preferred for Urdu/Arabic script)
            else:
                return 'ur'  # Default to Urdu for Arabic script (not Arabic language)
        
        # Hindi detection (Devanagari script) - map to Urdu instead
        hindi_pattern = re.compile(r'[\u0900-\u097F]')
        if hindi_pattern.search(sample):
            return 'ur'  # Map Hindi to Urdu voice
        
        # Chinese detection
        chinese_pattern = re.compile(r'[\u4E00-\u9FFF]')
        if chinese_pattern.search(sample):
            return 'zh'
        
        # Japanese detection
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')
        if japanese_pattern.search(sample):
            return 'ja'
        
        # Korean detection
        korean_pattern = re.compile(r'[\uAC00-\uD7AF]')
        if korean_pattern.search(sample):
            return 'ko'
        
        # Default to English
        return 'en'

    async def text_to_speech(self, text: str, output_path: Optional[str] = None, language: Optional[str] = None) -> str:
        """
        Convert text to speech using Edge TTS with automatic language detection.
        
        Args:
            text: Text to convert to speech
            output_path: Optional output file path
            language: Optional language code (ur, hi, ar, en, etc.). If None, auto-detects.
        
        Returns:
            Path to the generated audio file.
        """
        if not output_path:
            filename = f"response_{uuid.uuid4()}.mp3"
            output_path = filename

        try:
            # Detect language if not provided
            if not language:
                language = self._detect_language(text)
            
            # Get appropriate voice
            voice = self.voice_map.get(language, self.default_voice)
            
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            return output_path
        except Exception as e:
            print(f"Error during TTS generation: {e}")
            # Fallback to default voice
            try:
                communicate = edge_tts.Communicate(text, self.default_voice)
                await communicate.save(output_path)
                return output_path
            except Exception as fallback_error:
                print(f"Fallback TTS also failed: {fallback_error}")
                raise
