"""
Multilingual Support using Whisper, gTTS, Google Translate, and Gemini API
Supports voice input, text translation, and voice output
"""
import logging
from typing import Optional, Dict
import os
import tempfile

logger = logging.getLogger(__name__)

class MultilingualService:
    """Handle multilingual voice and text processing"""
    
    def __init__(self):
        self.whisper_model = None
        self.gemini_model = None
        self.google_translator = None
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'mr': 'Marathi',
            'pa': 'Punjabi'
        }
        
        # Initialize Google Translator
        try:
            from deep_translator import GoogleTranslator
            self.google_translator = GoogleTranslator
            logger.info("[OK] Google Translator initialized")
        except ImportError:
            logger.warning("[WARN] deep-translator not installed")
        
        # Initialize Gemini if API key provided
        if self.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                logger.info("[OK] Gemini API initialized for translation")
            except Exception as e:
                logger.warning(f"[WARN] Gemini initialization failed: {e}")
    
    def initialize_whisper(self):
        """Initialize Whisper for speech-to-text"""
        try:
            import whisper
            logger.info("[LOAD] Loading Whisper model for speech recognition...")
            self.whisper_model = whisper.load_model("base")  # base model for speed
            logger.info("[OK] Whisper model loaded successfully")
            return True
        except ImportError:
            logger.error("[ERROR] Whisper not installed. Install: pip install openai-whisper")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Failed to load Whisper: {e}")
            return False
    
    def initialize_translation(self):
        """Initialize mBERT/XLM-R for translation"""
        try:
            from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModel
            logger.info("[LOAD] Loading multilingual translation model...")
            
            # Use XLM-RoBERTa for multilingual understanding
            model_name = "xlm-roberta-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.translation_model = AutoModel.from_pretrained(model_name)
            
            logger.info("[OK] Translation model loaded successfully")
            return True
        except ImportError:
            logger.error("[ERROR] Transformers not installed. Install: pip install transformers")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Failed to load translation model: {e}")
            return False
    
    def transcribe_audio(self, audio_file_path: str, language: str = None) -> Dict:
        """
        Convert speech to text using Whisper
        
        Args:
            audio_file_path: Path to audio file
            language: Language code (optional, Whisper can auto-detect)
            
        Returns:
            Dict with text and detected language
        """
        if not self.whisper_model:
            if not self.initialize_whisper():
                return {"error": "Whisper model not available"}
        
        try:
            logger.info(f"[PROCESS] Transcribing audio file: {audio_file_path}")
            
            # Transcribe with language detection
            result = self.whisper_model.transcribe(
                audio_file_path,
                language=language,
                task="transcribe"
            )
            
            detected_language = result.get('language', 'en')
            text = result.get('text', '').strip()
            
            logger.info(f"[OK] Transcribed: '{text[:50]}...' (Language: {detected_language})")
            
            return {
                "text": text,
                "language": detected_language,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Transcription failed: {e}")
            return {"error": str(e), "success": False}
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Input text
            
        Returns:
            Language code
        """
        try:
            from langdetect import detect
            lang = detect(text)
            logger.info(f"[DETECT] Detected language: {lang}")
            return lang
        except Exception as e:
            logger.warning(f"[WARN] Language detection failed: {e}, defaulting to 'en'")
            return 'en'
    
    def text_to_speech(self, text: str, language: str = 'en') -> Optional[str]:
        """
        Convert text to speech using gTTS
        
        Args:
            text: Text to convert
            language: Language code
            
        Returns:
            Path to generated audio file
        """
        try:
            from gtts import gTTS
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_path = temp_file.name
            temp_file.close()
            
            logger.info(f"[PROCESS] Generating speech for text (Language: {language})")
            
            # Generate speech
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(temp_path)
            
            logger.info(f"[OK] Speech generated: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"[ERROR] Text-to-speech failed: {e}")
            return None
    
    def translate_query(self, text: str, source_lang: str, target_lang: str = 'en') -> str:
        """
        Translate text between languages using Google Translate or Gemini API
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        if source_lang == target_lang:
            return text
        
        # Try Google Translate first (free and fast)
        if self.google_translator:
            try:
                logger.info(f"[GOOGLE TRANSLATE] {source_lang} -> {target_lang}: '{text[:50]}...'")
                translator = self.google_translator(source=source_lang, target=target_lang)
                translated = translator.translate(text)
                logger.info(f"[OK] Google Translated: '{translated[:50]}...'")
                return translated
            except Exception as e:
                logger.warning(f"[WARN] Google Translate failed: {e}")
        
        # Try Gemini API if available and Google Translate failed
        if self.gemini_model:
            try:
                logger.info(f"[GEMINI] Translating {source_lang} -> {target_lang}")
                source_name = self.supported_languages.get(source_lang, source_lang)
                target_name = self.supported_languages.get(target_lang, target_lang)
                
                prompt = f"Translate the following {source_name} text to {target_name}. Only provide the translation, no explanations:\n\n{text}"
                response = self.gemini_model.generate_content(prompt)
                translated = response.text.strip()
                
                logger.info(f"[OK] Gemini Translated: '{translated[:50]}...'")
                return translated
            except Exception as e:
                logger.warning(f"[WARN] Gemini translation failed: {e}")
        
        # Fallback to translators library
        try:
            import translators as ts
            logger.info(f"[FALLBACK] Using translators library")
            translated = ts.translate_text(
                text,
                from_language=source_lang,
                to_language=target_lang,
                translator='google'
            )
            logger.info(f"[OK] Fallback translated: '{translated[:50]}...'")
            return translated
        except Exception as e:
            logger.error(f"[ERROR] All translation methods failed: {e}")
            return text
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.supported_languages


# Global instance
multilingual_service = MultilingualService()
