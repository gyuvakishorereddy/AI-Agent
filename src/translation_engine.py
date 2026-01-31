"""
Multilingual Translation Engine for KARE AI
Supports automatic language detection and response translation
"""

import logging
from typing import Dict, Optional, Tuple
from langdetect import detect, LangDetectException, detect_langs
import re

logger = logging.getLogger(__name__)

# Language codes mapping (Required languages only)
LANGUAGE_CODES = {
    'en': 'English',
    'ta': 'Tamil',
    'te': 'Telugu',
    'hi': 'Hindi',
    'kn': 'Kannada',
    'ml': 'Malayalam',
}

# Language detection with script recognition
SCRIPT_RANGES = {
    'ta': (0x0B80, 0x0BFF),  # Tamil
    'te': (0x0C00, 0x0C7F),  # Telugu
    'hi': (0x0900, 0x097F),  # Devanagari (Hindi)
    'kn': (0x0C80, 0x0CFF),  # Kannada
    'ml': (0x0D00, 0x0D7F),  # Malayalam
}

# Language-specific greeting responses (5 languages only)
MULTILINGUAL_RESPONSES = {
    'hi': {
        'greeting': "Namaste! 🙏 KARE AI Assistant mein aapka swagat hai!",
        'help_intro': "Main aapke liye madad kar sakta hoon:",
        'ask_more': "Aur kuch puchna hai? Mujhe batayein! 😊",
        'no_result': "Maaf kijiye, mujhe aapke sawal ka jawab nahi mila. Kripya doosra sawal puchiye.",
        'error': "Maaf kijiye, kuch error ho gaya. Kripya dobara try kijiye.",
    },
    'te': {
        'greeting': "Namaskaram! 🙏 KARE AI Assistant ki ochcharu!",
        'help_intro': "Nenu meemanaku sahayam cheyya vachchu:",
        'ask_more': "Inka emaina adigithe cheppandi! 😊",
        'no_result': "Kshaminchanadi, nenu mee prashnaku samacharam kanaledu.",
        'error': "Kshaminchanadi, konni tappu jarigindi. Dayachesi marala prayatinchandi.",
    },
    'ta': {
        'greeting': "Vanakkam! 🙏 KARE AI Assistant ku varaverukkirom!",
        'help_intro': "Naan ungaluku uthavi seya mudiyum:",
        'ask_more': "Innum ethavadhu ketkalama? Sollunga! 😊",
        'no_result': "Mannikavum, naan ungal kelviyin badhilai kandupidikka mudiyavillai.",
        'error': "Mannikavum, oru pizhai nadandhadhu. Punaha muyarchi seyyavum.",
    },
    'kn': {
        'greeting': "Namaskara! 🙏 KARE AI Assistant ge swaagata!",
        'help_intro': "Naanu nimge sahaya maadabahudu:",
        'ask_more': "Innu yaavudadaru kelabeku? Heli! 😊",
        'no_result': "Kshamisi, naanu nimma prashnege maahiti kanaledu.",
        'error': "Kshamisi, ondu tappu aayitu. Dayavittu maathondhe prayatnisi.",
    },
    'ml': {
        'greeting': "Namaskaram! 🙏 KARE AI Assistant lek swagatham!",
        'help_intro': "Njan ningalod sahayikkan kazhiyum:",
        'ask_more': "Innenthengilum chodikkanam? Parayoo! 😊",
        'no_result': "Kshamikkanam, ningalude chodhyathin samaadhaanam kittiyilla.",
        'error': "Kshamikkanam, onnu tetattu sambhavichhu. Dayavaai maarthay prayathikku.",
    },
    'en': {
        'greeting': "Hi there! 👋 Welcome to KARE AI Assistant!",
        'help_intro': "I can help you with:",
        'ask_more': "Need more details? Just ask! 😊",
        'no_result': "I couldn't find information about that. Please try another question.",
        'error': "Sorry, I encountered an error. Please try again.",
    }
}

class LanguageDetector:
    """Detect language from text using multiple methods"""
    
    # Romanized language patterns for common phrases (Hinglish, Tenglish, Tanglish, etc.)
    ROMANIZED_PATTERNS = {
        'hi': [
            # Hindi greetings
            'namaste', 'namaskar', 'namaskaar', 'pranam', 'dhanyavad', 'shukriya',
            # Common Hinglish words - pronouns
            'mai', 'main', 'mein', 'aap', 'aapka', 'aapki', 'aapke',
            'tum', 'tumhara', 'tumhari', 'tumhare', 'mera', 'meri', 'mere',
            'hamara', 'hamari', 'hamare', 'unka', 'unki', 'unke',
            # Questions
            'kya', 'kyu', 'kyun', 'kyunki', 'kab', 'kaha', 'kahan',
            'kaun', 'kaise', 'kaisa', 'kaisi', 'kaunsa', 'kitna', 'kitne', 'kitni',
            # Verbs (common conjugations)
            'hai', 'hain', 'ho', 'hoon', 'hu', 'tha', 'thi', 'the',
            'hoga', 'hogi', 'honge', 'hua', 'hui', 'huye', 'hue',
            'karo', 'kare', 'karenge', 'karunga', 'karungi', 'kiya', 'kiye', 'karke', 'karne',
            'jao', 'jaye', 'jayenge', 'gaya', 'gayi', 'gaye', 'jaakar', 'jaane',
            'aao', 'aaye', 'aayenge', 'aaya', 'aayi', 'aaye', 'aakar', 'aane',
            'dekho', 'dekha', 'dekhe', 'dekhiye', 'dekhna', 'dekhenge',
            'batao', 'bataye', 'bataiye', 'batana', 'bataya', 'batayenge',
            'suniye', 'suno', 'suna', 'sunna', 'sunenge', 'sunte',
            'chalo', 'chalte', 'chalenge', 'chala', 'chali', 'chale',
            # Common words
            'haan', 'han', 'nahin', 'nahi', 'nhi', 'mat', 'mujhe', 'mujhko',
            'tumhe', 'tumko', 'use', 'usse', 'inhe', 'unhe', 'isko', 'usko',
            'acha', 'achha', 'accha', 'bahut', 'bohot', 'boht', 'thoda', 'jyada', 'zyada',
            'kuch', 'kuchh', 'kucch', 'sabhi', 'sab', 'yeh', 'ye', 'woh', 'wo',
            'abhi', 'ab', 'phir', 'fir', 'bhi', 'bhe', 'aur', 'ya', 'yaa',
        ],
        'ta': [
            # Tamil greetings
            'vanakkam', 'vanakam', 'vannakkam', 'vanakaam', 'nandri', 'nanri', 'romba', 'nandri',
            # Common Tanglish words - questions
            'enna', 'yenna', 'yen', 'epdi', 'eppadi', 'yeppadi', 'eppudi',
            'eppo', 'yeppo', 'eppozhudu', 'enga', 'yenga', 'engada', 'yaar', 'yaaru', 'evaru',
            # Pronouns
            'naan', 'nan', 'naanu', 'nee', 'neenga', 'nenga', 'nigga',
            'naanga', 'naanga', 'unga', 'ungal', 'ungala', 'enga', 'engal',
            # Tamil verbs - to be/have
            'irukku', 'iruku', 'irukkum', 'irukkudhu', 'irundha', 'irundhuchi', 'irundhuchu',
            'illa', 'illai', 'illaya', 'illaiya', 'illaye', 'ille',
            # Common verbs
            'panna', 'pannunga', 'pannum', 'pannurathu', 'pannanum', 'pannikka', 'panniruken',
            'sollu', 'sollunga', 'sollanum', 'solla', 'sonna', 'sonninga', 'solranga', 'solluvanga',
            'venum', 'vendum', 'venam', 'venaam', 'vendaam', 'venumnu', 'venumna', 'venumaanu',
            'kudukka', 'kudu', 'kudunga', 'kuduthuru', 'kodukka', 'kudukkanum',
            # Actions
            'poga', 'pogalam', 'pogalaam', 'pochu', 'poidum', 'poittu', 'poirukken',
            'vaanga', 'vanga', 'varum', 'varudhu', 'varuma', 'varuvanga', 'vanthuru',
            'sapadu', 'saapadu', 'saapidanum', 'sapidanum', 'saapdu', 'saapten',
            'kudikka', 'kudi', 'kudichitu', 'kudichu',
            'padichu', 'padikka', 'padichitu', 'padikkanum', 'ezhutu', 'ezhuthu',
            # Common expressions
            'achu', 'aachu', 'ayiduchu', 'aachu', 'agum', 'aagum', 'aarum',
            'thaan', 'taan', 'dhan', 'than', 'mattum', 'maththum', 'mattu', 'mattumthan',
            'kooda', 'kuda', 'koodi', 'thana', 'thanam', 'thaana',
            'seiyanum', 'seyya', 'seyyanum', 'seiyalaam', 'senji', 'senjitu',
        ],
        'te': [
            # Telugu greetings
            'namaskaram', 'namaskaramandi', 'namaskaaram', 'dhanyavadamulu', 'dhanyavadalu', 'dhanyavadalu',
            # Common Tenglish words - questions
            'ela', 'elaa', 'yela', 'ila', 'ilaa', 'enti', 'yenti', 'emiti', 'emi', 'yemi',
            'ekkada', 'yakkada', 'ekkadiki', 'ekkadundi', 'ekkadinundi',
            'eppudu', 'epudu', 'yeppudu', 'eppudaina', 'eppudayna',
            'endhuku', 'enduku', 'yenduku', 'endukante', 'endukanti', 'yendukante',
            # Pronouns
            'nenu', 'nenu', 'nuvvu', 'nuvu', 'nuvvu', 'meeru', 'miru',
            'memu', 'manam', 'manamu', 'vaaru', 'evaru', 'evaaru', 'evarandi',
            # Telugu verbs - to be/have
            'undi', 'undhi', 'vundi', 'unnadi', 'vunnadi', 'untundi', 'untaadu', 'untaru', 'unnaaru',
            'ledu', 'ledhu', 'leru', 'kaadu', 'kadu',
            'ayyi', 'ayyindi', 'aindi', 'ayyindhi', 'ayyaru', 'ayyaaru',
            # Common verbs
            'cheppandi', 'cheppu', 'cheppanu', 'cheppara', 'chepthe', 'cheptunna', 'cheppali',
            'chestunna', 'chesanu', 'chesa', 'cheyyali', 'cheyali', 'cheyandi', 'chesara',
            'cheyadaniki', 'cheyyadaniki', 'cheyadaaniki', 'cheyyataniki', 'cheyyalani',
            # Actions
            'vacchu', 'vachchu', 'vachu', 'vacchindi', 'vaccharu', 'vachanu', 'vachesanu',
            'vellu', 'vellandi', 'velthanu', 'veltanu', 'velpoya', 'vellipoya',
            'ivvandi', 'ivvu', 'ivvali', 'ivvaali', 'ichindi', 'ichcharu',
            'teesko', 'teeskondi', 'teeskovali', 'tisuko', 'teesukoni', 'teeskunna',
            'tinu', 'tinandi', 'tintanu', 'tinna', 'tinnanu', 'tinali',
            'avvali', 'avvaali', 'avvandi', 'avuthanu', 'avuthunna', 'ayyanu',
            # Telugu postpositions
            'lo', 'lone', 'loki', 'nunchi', 'nundi', 'daggara', 'daggira',
            'kosam', 'koosam', 'valla', 'valle', 'tho', 'thone',
            # Common expressions
            'chalu', 'chaalu', 'calu', 'chalunu', 'chaalu', 'calu',
            'okkati', 'okati', 'oka', 'okka', 'rendu', 'rendhu', 'moodu', 'nalugu',
            'manchi', 'manchidi', 'manchidhi', 'manchoddhi', 'manchiga',
            'anthe', 'ante', 'antey', 'anthey', 'ayna', 'ayina', 'kuda', 'kooda', 'kuda',
            'gurinchi', 'guurinchi', 'gurchi', 'gurinche', 'gurinchi',
            'unnav', 'unnavu', 'unnaavu', 'unnara', 'unnaru', 'unnaara',
        ],
        'kn': [
            # Kannada greetings
            'namaskara', 'namaskaara', 'namaste', 'dhanyavaadagalu', 'dhanyavada', 'dhanyavaada',
            # Common Kannada words - questions
            'enu', 'yenu', 'enta', 'hege', 'hegiddeeri', 'heege', 'hegide', 'hegiddira',
            'yaavaga', 'yavaga', 'elli', 'yelli', 'ellidira', 'yellidira',
            'yake', 'yaake', 'yaakaandre', 'yaakandre',
            # Pronouns
            'naanu', 'nanu', 'naavu', 'neevu', 'nivu', 'neenu', 'niivu',
            'avaru', 'avaaru', 'ivaru', 'eevaru', 'yaaru', 'yaru',
            # Kannada verbs - to be/have
            'ide', 'idhe', 'idhe', 'idare', 'iddare', 'iddaare',
            'illa', 'ilve', 'illa', 'illavo', 'illappa',
            'agutte', 'aaagutte', 'aagide', 'aagutta', 'aagutthe',
            'aytu', 'aaytu', 'aayitu', 'aayithu',
            # Common verbs
            'maadu', 'maadi', 'maadbeku', 'maadona', 'maadidiri', 'maadtini', 'maadtare',
            'helu', 'heli', 'helidiri', 'helabeku', 'helthare', 'heltini', 'heltini',
            'kodu', 'kodi', 'kodri', 'kodbeku', 'kodona', 'kodtini', 'kodtare',
            'togo', 'togoli', 'togobeku', 'togona', 'togotini', 'togotare',
            'bandu', 'bandide', 'barthini', 'bartheera', 'barodu', 'bartini', 'bartare',
            # Common expressions
            'beku', 'bekilla', 'bekaadre', 'beda', 'beku', 'beeku',
            'houdu', 'houda', 'aalla', 'alla', 'allo',
            'sari', 'saari', 'sariyaagi', 'chennagi', 'chennagide',
            'thumba', 'thumma', 'thumbane', 'eshtu', 'estu', 'yeshtu',
        ],
        'ml': [
            # Malayalam greetings
            'namaskaram', 'namaskaaram', 'nanni', 'nanmayulle', 'nanni',
            # Common Malayalam words - questions
            'enthu', 'enthaa', 'entha', 'enthaanu', 'enthaan',
            'engane', 'enganeya', 'enganund', 'enganeund', 'enganaanu',
            'evide', 'evideyaa', 'evideyaanu', 'evideyanu',
            'eppo', 'eppol', 'eppozhaanu', 'eppozha',
            'enthina', 'enthukond', 'enthukondaanu', 'endinu',
            # Pronouns
            'njan', 'njaan', 'njangal', 'njaangal',
            'ningal', 'ningalu', 'ningalkku', 'thaan', 'thangal',
            'avan', 'aval', 'avar', 'avaru', 'aaru', 'aaranu',
            # Malayalam verbs - to be/have
            'und', 'undu', 'undallo', 'undayirunnu', 'undaayi',
            'illa', 'illallo', 'illaayirunnu', 'illayi',
            'aakum', 'aavum', 'aayi', 'aayallo', 'aayirikum',
            # Common verbs
            'cheyyuka', 'cheyyu', 'cheyyaan', 'cheyyanda', 'cheyth', 'cheythaal',
            'parayuka', 'paranju', 'parayaan', 'parayaamo', 'parayum', 'parayunnu',
            'thaa', 'tharanam', 'tharaam', 'tharumo', 'tharan', 'tharunnath',
            'vannu', 'varum', 'varaan', 'varaamallo', 'varatte', 'varanam',
            'pokuka', 'poyi', 'pokaan', 'pokaam', 'pokatte', 'pokanam',
            # Common expressions
            'venam', 'venda', 'vendaa', 'venamenkil', 'venamallo',
            'unde', 'undennu', 'undenna', 'undello',
            'sheri', 'sheriyan', 'nannaayi', 'nannayi', 'valare', 'valarthaan',
            'kurachu', 'kurach', 'kuranju', 'korachu',
        ],
    }
    
    @staticmethod
    def detect_by_script(text: str) -> Optional[str]:
        """Detect language by checking Unicode script ranges"""
        for lang, (start, end) in SCRIPT_RANGES.items():
            if any(start <= ord(char) <= end for char in text):
                return lang
        return None
    
    @staticmethod
    def detect_by_romanized_pattern(text: str) -> Optional[str]:
        """Detect romanized language by checking common word patterns"""
        text_lower = text.lower().strip()
        words = re.split(r'[^a-z]+', text_lower)
        
        # Filter out very short words and common English words
        common_english = {
            'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
            'can', 'may', 'might', 'must', 'shall',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'this', 'that', 'these', 'those',
            'a', 'an', 'and', 'or', 'but', 'if', 'because',
            'to', 'from', 'in', 'on', 'at', 'by', 'for', 'with', 'about',
            'what', 'where', 'when', 'who', 'which', 'how', 'why',
        }
        
        # Count English words
        english_word_count = sum(1 for w in words if w in common_english)
        non_english_words = [w for w in words if w not in common_english and len(w) >= 2]
        
        # If mostly English words (> 60%), likely English
        if len(words) > 0 and english_word_count / len(words) > 0.6:
            logger.debug(f"Mostly English words detected: {english_word_count}/{len(words)}")
            return 'en'
        
        # Special handling for single-word greetings
        if len(words) == 1 or (len(words) == 2 and len(text_lower) < 15):
            greeting_matches = {
                'ta': ['vanakkam', 'vanakam', 'vannakkam'],
                'te': ['namaskaramandi'],  # Telugu-specific longer form
                'hi': ['namaste', 'namaskaar'],
                'kn': ['namaskara', 'namaskaara'],
                'ml': [],  # Malayalam 'namaskaram' conflicts with Telugu, handle in patterns
            }
            for lang, greetings in greeting_matches.items():
                if any(greeting in text_lower for greeting in greetings):
                    logger.debug(f"Single-word greeting detected: {text_lower} -> {lang}")
                    return lang
        
        # Score each language based on matching patterns (use non-English words only)
        scores = {lang: 0 for lang in LanguageDetector.ROMANIZED_PATTERNS}
        
        for word in non_english_words:
            for lang, patterns in LanguageDetector.ROMANIZED_PATTERNS.items():
                # Exact word match (higher priority)
                if word in patterns:
                    scores[lang] += 3
                # Substring match (lower priority)
                elif any(pattern in word or word in pattern for pattern in patterns):
                    scores[lang] += 1
        
        # Return language with highest score (if any)
        if max(scores.values()) > 0:
            best_lang = max(scores, key=scores.get)
            # Need at least one strong match or multiple weak matches
            if scores[best_lang] >= 3 or scores[best_lang] >= 2:
                logger.debug(f"Romanized pattern scores: {scores}, selected: {best_lang}")
                return best_lang
        
        return None
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect language with fallback methods
        Returns language code (en, hi, te, ta, etc.)
        """
        if not text or len(text.strip()) < 2:
            return 'en'
        
        try:
            # First, try script detection for Indian scripts (most reliable)
            script_lang = LanguageDetector.detect_by_script(text)
            if script_lang:
                logger.info(f"✅ Language detected by script: {script_lang}")
                return script_lang
            
            # Second, try romanized pattern matching
            romanized_lang = LanguageDetector.detect_by_romanized_pattern(text)
            if romanized_lang:
                logger.info(f"✅ Language detected by romanized pattern: {romanized_lang}")
                return romanized_lang
            
            # Third, try langdetect library
            lang = detect(text)
            
            # Map to supported languages
            if lang in LANGUAGE_CODES:
                logger.info(f"✅ Language detected by langdetect: {lang}")
                return lang
            
            # For unsupported languages, return English
            logger.info(f"⚠️ Language {lang} not in supported list, returning English")
            return 'en'
            
        except LangDetectException:
            # If detection fails, return English
            logger.warning("⚠️ LangDetect exception, returning English")
            return 'en'
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return 'en'
    
    @staticmethod
    def get_language_name(lang_code: str) -> str:
        """Get language name from code"""
        return LANGUAGE_CODES.get(lang_code, 'English')


class ResponseTranslator:
    """Translate responses to detected language"""
    
    def __init__(self):
        """Initialize translation engine"""
        try:
            import translators as ts
            self.ts = ts
            self.translation_available = True
        except ImportError:
            logger.warning("⚠️ Translators library not installed. Using fallback responses only.")
            self.ts = None
            self.translation_available = False
    
    @staticmethod
    def extract_text_for_translation(response: str) -> str:
        """Extract main text from markdown response"""
        # Remove markdown formatting but keep content
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)  # Remove bold
        text = re.sub(r'_([^_]+)_', r'\1', response)  # Remove italic
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Remove links
        return text
    
    def translate_response(self, response: str, target_lang: str) -> str:
        """
        Translate response to target language
        Falls back to greeting templates for common topics
        """
        
        if target_lang == 'en':
            return response
        
        # For Indian languages, use predefined responses for common queries
        if target_lang in MULTILINGUAL_RESPONSES:
            response_templates = MULTILINGUAL_RESPONSES[target_lang]
            
            # Detect query topic and return templated response if available
            lower_response = response.lower()
            
            if 'greeting' in lower_response or 'welcome' in lower_response:
                return response_templates['greeting']
            
            # Try to preserve structure and just translate key sections
            try:
                if self.translation_available and self.ts:
                    return self._translate_with_library(response, target_lang)
                else:
                    logger.info(f"Using fallback responses for {target_lang}")
                    return response  # Return English if translation not available
            except Exception as e:
                logger.error(f"Translation failed: {e}")
                return response
        
        return response
    
    def _translate_with_library(self, text: str, target_lang: str) -> str:
        """Translate using translators library"""
        try:
            if self.ts is None:
                return text
            
            # Split into sentences for better translation
            sentences = re.split(r'(?<=[.!?])\s+', text)
            translated_parts = []
            
            for sentence in sentences:
                if len(sentence.strip()) < 2:
                    translated_parts.append(sentence)
                    continue
                
                try:
                    # Use Google Translate via translators
                    translated = self.ts.translate_text(
                        query_text=sentence,
                        from_language='auto',
                        to_language=target_lang
                    )
                    translated_parts.append(translated)
                except Exception as e:
                    logger.warning(f"Sentence translation failed: {e}")
                    translated_parts.append(sentence)
            
            return ' '.join(translated_parts)
        
        except Exception as e:
            logger.error(f"Translation library error: {e}")
            return text


class MultilingualResponseGenerator:
    """Generate responses in detected language"""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.translator = ResponseTranslator()
    
    def process_response(self, query: str, response: str) -> Tuple[str, str]:
        """
        Process response based on detected query language
        
        Returns:
            Tuple of (translated_response, detected_language_code)
        """
        # Detect language from query
        detected_lang = self.language_detector.detect_language(query)
        logger.info(f"🌍 Detected language: {detected_lang} ({LANGUAGE_CODES.get(detected_lang, 'Unknown')})")
        
        # Translate response to detected language
        translated = self.translator.translate_response(response, detected_lang)
        
        return translated, detected_lang
    
    def get_greeting(self, detected_lang: str) -> str:
        """Get greeting in detected language"""
        return MULTILINGUAL_RESPONSES.get(
            detected_lang, 
            MULTILINGUAL_RESPONSES['en']
        )['greeting']
    
    def add_language_context(self, response: str, lang_code: str) -> str:
        """Add language context to response"""
        if lang_code != 'en':
            lang_name = LANGUAGE_CODES.get(lang_code, lang_code.upper())
            context = f"*[Responded in {lang_name}]*\n\n"
            return context + response
        return response


# Global instance
multilingual_engine = MultilingualResponseGenerator()


def detect_language(text: str) -> str:
    """Utility function to detect language"""
    return LanguageDetector.detect_language(text)


def translate_response(response: str, target_lang: str) -> str:
    """Utility function to translate response"""
    translator = ResponseTranslator()
    return translator.translate_response(response, target_lang)


def process_multilingual_response(query: str, response: str) -> Tuple[str, str]:
    """Utility function to process response in query language"""
    return multilingual_engine.process_response(query, response)
