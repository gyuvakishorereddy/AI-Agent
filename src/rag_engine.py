"""
RAG Response Generator with FAISS Vector Store + Gemma2-9B Q4
Integrates vector similarity search with LLM generation for intelligent responses
Supports: English, Hindi, Telugu, Tamil, Kannada, Malayalam
         + Hinglish, Tinglish (Tenglish), Tanglish, and mixed-language queries
"""

import re
import logging
from pathlib import Path
from typing import Optional, List, Dict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from vector_store import VectorStoreManager
from gemma2_llm import Gemma2LLM

# deep-translator: free Google Translate, no API key required
try:
    from deep_translator import GoogleTranslator as _GoogleTranslator
    _DEEP_TRANSLATOR_AVAILABLE = True
except Exception:
    _DEEP_TRANSLATOR_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Relevance threshold: cosine similarity below this → not relevant to KARE
# With all-MiniLM-L6-v2 + proper cosine sim, relevant hits are typically >0.35
# ---------------------------------------------------------------------------
RELEVANCE_THRESHOLD = 0.35


class RAGResponseGenerator:
    """Generate responses using RAG (Retrieval-Augmented Generation) with FAISS + Gemma2"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vector_store_path: str = "faiss_index",
        data_dir: str = "data",
        use_llm: bool = True
    ):
        self.use_llm = use_llm
        self.llm_available = False
        
        # Initialize vector store
        logger.info("🔄 Initializing RAG system...")
        
        self.vector_store = VectorStoreManager(
            data_dir=data_dir,
            vector_store_path=vector_store_path
        )
        
        if not self.vector_store.load_vector_store():
            logger.warning("⚠️ Vector store not found. Please run build_vector_store.py first.")
        
        # Initialize LLM (if requested)
        self.llm = None
        if use_llm:
            try:
                logger.info("🤖 Initializing Gemma2-9B Q4 LLM...")
                self.llm = Gemma2LLM(
                    model_path=model_path,
                    n_ctx=4096,
                    n_gpu_layers=0,
                    n_threads=4,
                    temperature=0.7,
                    max_tokens=512,
                    verbose=False
                )
                self.llm_available = self.llm.is_initialized
                if self.llm_available:
                    logger.info("✅ RAG system ready (Vector Store + Gemma2 LLM)")
                else:
                    logger.warning("⚠️ Gemma2 not initialized. Using template responses.")
            except Exception as e:
                logger.error(f"❌ Failed to initialize LLM: {e}")
                self.llm_available = False
        else:
            logger.info("✅ RAG system ready (Vector Store only - template mode)")

        # Log translation engine status
        if _DEEP_TRANSLATOR_AVAILABLE:
            logger.info("✅ deep-translator ready (free multilingual, no API key)")
        else:
            logger.warning("⚠️ deep-translator not installed; responses will be in English")

    # ------------------------------------------------------------------
    # Free translation helpers (deep-translator, no API key)
    # ------------------------------------------------------------------
    def _translate_text(self, text: str, source: str, target: str) -> str:
        """Translate text using deep-translator (Google Translate, free).
        Preserves URLs, emails, and KARE-specific proper nouns."""
        if not _DEEP_TRANSLATOR_AVAILABLE or source == target:
            return text
        if not text or not text.strip():
            return text

        # Extract and protect URLs / emails before translation
        import re as _re
        placeholders = {}
        protected = text

        # Protect URLs
        urls = _re.findall(r'https?://\S+', text)
        for i, url in enumerate(urls):
            ph = f'URLPH{i}URLPH'
            placeholders[ph] = url
            protected = protected.replace(url, ph, 1)

        # Protect emails
        emails = _re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', protected)
        for i, email in enumerate(emails):
            ph = f'EMAILPH{i}EMAILPH'
            placeholders[ph] = email
            protected = protected.replace(email, ph, 1)

        try:
            # deep-translator has a 5000 char limit per call; split if needed
            if len(protected) <= 4900:
                translated = _GoogleTranslator(source=source, target=target).translate(protected)
            else:
                # Split on double-newlines to preserve formatting
                parts = protected.split('\n\n')
                translated_parts = []
                for part in parts:
                    if part.strip():
                        t = _GoogleTranslator(source=source, target=target).translate(part)
                        translated_parts.append(t if t else part)
                    else:
                        translated_parts.append(part)
                translated = '\n\n'.join(translated_parts)

            if not translated:
                return text

            # Restore placeholders
            for ph, original in placeholders.items():
                translated = translated.replace(ph, original)

            return translated
        except Exception as e:
            logger.warning(f"⚠️ Translation failed ({source}→{target}): {e}")
            return text

    # ------------------------------------------------------------------
    # Language Translation Helper
    # ------------------------------------------------------------------
    def _translate_to_english(self, query: str, language: str) -> str:
        """Translate non-English query to English for better FAISS matching."""
        if language == 'en' or not query.strip():
            return query

        # Keyword-level fallback maps (handles cases where the translator struggles)
        _KEYWORD_MAPS = {
            'ta': {
                'விடுதி': 'hostel', 'கட்டணம்': 'fees', 'வலைத்தளம்': 'website',
                'சேர்க்கை': 'admission', 'முன்பதிவு': 'booking',
                'திட்டங்கள்': 'programs', 'துறை': 'department',
                'வேலைவாய்ப்பு': 'placement', 'போக்குவரத்து': 'transport',
                'பஸ்': 'bus', 'உணவு': 'food', 'உணவகம்': 'mess',
                'வசதிகள்': 'facilities', 'நூலகம்': 'library',
                'ஆய்வு': 'research', 'உதவித்தொகை': 'scholarship',
                'பற்றி': 'about', 'என்ன': 'what', 'எங்கே': 'where',
                'எப்படி': 'how', 'இருக்கிறது': '', 'செய்வதற்கான': 'for',
            },
            'te': {
                'హాస్టెల్': 'hostel', 'ఫీజు': 'fees', 'వెబ్‌సైట్': 'website',
                'ప్రవేశం': 'admission', 'బుకింగ్': 'booking',
                'కార్యక్రమాలు': 'programs', 'శాఖ': 'department',
                'ప్లేస్‌మెంట్': 'placement', 'రవాణా': 'transport',
                'స్కాలర్‌షిప్': 'scholarship',
            },
            'hi': {
                'हॉस्टल': 'hostel', 'फीस': 'fees', 'वेबसाइट': 'website',
                'प्रवेश': 'admission', 'बुकिंग': 'booking',
                'कार्यक्रम': 'programs', 'विभाग': 'department',
                'छात्रवृत्ति': 'scholarship', 'परिवहन': 'transport',
            },
        }

        # Apply keyword substitution first (reliable)
        result = query
        for native, english in _KEYWORD_MAPS.get(language, {}).items():
            result = result.replace(native, f' {english} ')
        result = ' '.join(result.split()).strip()

        # If substitution changed the query meaningfully, use it
        if result != query and any(c.isascii() and c.isalpha() for c in result):
            logger.info(f"📝 Keyword translation ({language}→en): {result[:80]}")
            return result

        # Try deep-translator with auto-detect (more reliable than explicit lang code)
        if _DEEP_TRANSLATOR_AVAILABLE:
            try:
                translated = _GoogleTranslator(source='auto', target='en').translate(query)
                if translated and translated.strip() and translated != query:
                    logger.info(f"📝 deep-translator ({language}→en): {translated[:80]}")
                    return translated
            except Exception as e:
                logger.debug(f"deep-translator query translation failed: {e}")

        return result if result != query else query

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def generate_response(self, query: str, language: str = 'en', top_k: int = 5) -> str:
        """
        Generate response using RAG pipeline.
        1. Greeting / conversational check
        2. FAISS search (with translation if needed)
        3. Relevance filtering
        4. Response generation (LLM or template)
        """
        logger.info(f"🎯 Query: {query[:80]}... (Language: {language})")
        
        # Step 0: Check for greetings / casual conversation
        if self._is_greeting(query):
            logger.info("👋 Greeting detected")
            return self._get_greeting_response(language)
        
        if self._is_conversational(query):
            logger.info("💬 Conversational query detected")
            return self._get_conversational_response(query, language)
        
        # Step 1: Translate query to English if needed (for better vector matching)
        search_query = query
        if language != 'en':
            search_query = self._translate_to_english(query, language)
            if search_query != query:
                logger.info(f"📝 Translated query: {search_query}")
        
        # Dynamically increase top_k for comprehensive queries
        dynamic_top_k = top_k
        query_lower = query.lower()
        if any(kw in query_lower for kw in ['document', 'submit', 'eligibility', 'criteria', 'require', 'procedure', 'process', 'steps', 'how to', 'hostel alloc', 'fee', 'program', 'branch', 'specializ']):
            dynamic_top_k = 12  # Get more context for comprehensive answers
        
        # Step 2: Retrieve relevant context from vector store
        context_chunks = self.vector_store.search(search_query, top_k=dynamic_top_k)
        
        if not context_chunks:
            logger.warning("⚠️ No results from FAISS")
            return self._get_no_info_response(query, language)
        
        # Step 2: Filter by relevance threshold
        relevant_chunks = [
            c for c in context_chunks
            if c.get('similarity_score', 0) >= RELEVANCE_THRESHOLD
        ]
        
        if not relevant_chunks:
            best_score = context_chunks[0].get('similarity_score', 0)
            logger.warning(
                f"⚠️ No chunks above threshold {RELEVANCE_THRESHOLD}. "
                f"Best score: {best_score:.3f} for query: '{query}'"
            )
            return self._get_no_info_response(query, language)
        
        logger.info(
            f"📚 {len(relevant_chunks)}/{len(context_chunks)} chunks above threshold "
            f"(best: {relevant_chunks[0].get('similarity_score', 0):.3f})"
        )
        
        # Step 3: Format context
        context_text = self._format_context(relevant_chunks)
        
        # Step 4: Generate response (always in English via template/LLM)
        if self.llm_available and self.llm:
            response = self._generate_with_llm(query, context_text, language)
        else:
            response = self._generate_template_response(query, relevant_chunks, language)

        # Step 5: Translate full response to target language (free, no API key)
        if language != 'en' and _DEEP_TRANSLATOR_AVAILABLE:
            logger.info(f"🌐 Translating response en→{language} ({len(response)} chars)...")
            response = self._translate_text(response, source='en', target=language)
            logger.info(f"✅ Translation done → {language} ({len(response)} chars)")

        return response

    # ------------------------------------------------------------------
    # Greeting & conversational detection
    # ------------------------------------------------------------------
    def _is_greeting(self, query: str) -> bool:
        """Detect greetings in English, Hindi, Telugu, Tamil + Romanized variants"""
        q = query.lower().strip().rstrip('?!.,')
        
        # Exact / near-exact short greetings
        short_greetings = {
            # English
            'hi', 'hello', 'hey', 'hii', 'hiii', 'yo', 'sup', 'hola',
            'good morning', 'good afternoon', 'good evening', 'good night',
            'gm', 'gn',
            # Hindi / Hinglish
            'namaste', 'namaskar', 'namasthe', 'namaskaram', 'pranam',
            # Telugu / Tinglish
            'namaskaram', 'baagunnara', 'bagunnara',
            # Tamil / Tanglish
            'vanakkam', 'vannakam',
            # Kannada
            'namaskara',
            # Malayalam
            'namaskkaram',
        }
        
        if q in short_greetings:
            return True
        
        # Pattern matches (allow surrounding words)
        greeting_patterns = [
            r'\b(hello|hi|hey|hii+)\b',
            r'\bgood\s+(morning|afternoon|evening|night)\b',
            r'\b(namaste|namaskar|namasthe|namaskaram|vanakkam)\b',
            r'\b(pranam|vannakam|namaskara)\b',
        ]
        
        for pattern in greeting_patterns:
            if re.search(pattern, q):
                # Make sure it's not also asking about KARE topics
                if not self._has_kare_keywords(q):
                    return True
        
        return False

    def _is_conversational(self, query: str) -> bool:
        """Detect casual / conversational queries that are NOT about KARE"""
        q = query.lower().strip().rstrip('?!.,')
        
        # If query contains KARE-specific keywords, it's NOT just conversational
        if self._has_kare_keywords(q):
            return False
        
        conversational_patterns = [
            # English
            r'\bhow\s+are\s+you\b', r'\bwho\s+are\s+you\b', r'\bwhat\s+are\s+you\b',
            r'\bwhat\s+is\s+your\s+name\b', r"\bwhat'?s\s+your\s+name\b",
            r'\bwhat\s+can\s+you\s+do\b', r'\bhow\s+do\s+you\s+do\b',
            r'\bwhat\s+are\s+you\s+doing\b', r'\bwhat\s+you\s+doing\b',
            r'\bwhat\s+do\s+you\s+do\b', r'\bwhat\s+r\s+u\s+doing\b',
            r'\bwhat\'?s\s+up\b', r'\bwhats\s+up\b', r'\bwassup\b',
            r'\bnice\s+to\s+meet\b', r'\bthank\s*(you|s)\b', r'\bthanks\b',
            r'\bbye\b', r'\bgoodbye\b', r'\bsee\s+you\b', r'\bsee\s+ya\b',
            r'\bok(ay)?\b$', r'\balright\b$', r'\bsure\b$', r'\bfine\b$',
            r'\bhaha\b', r'\blol\b', r'\bhehe\b',
            
            # Hindi / Hinglish
            r'\bkaise\s+h(o|ai|ain)\b', r'\bkaisa\s+hai\b', r'\bkaisi\s+h(o|ai)\b',
            r'\bkya\s+hal\s+hai\b', r'\bkya\s+haal\s+hai\b',
            r'\baap\s+kaise\s+h(o|ain)\b', r'\btum\s+kaise\s+ho\b',
            r'\bkya\s+kart[aie]\s*h(ai|o)\b',           # kya karta hai, kya karti ho
            r'\bkya\s+kart[aie]\b',                       # kya karta (without hai)
            r'\bkya\s+kar\s+rah[aei]\s*h(o|ai)\b',       # kya kar raha ho
            r'\btum\s+kya\s+kart[aie]\b',                 # tum kya karta
            r'\baap\s+kya\s+kart[eai]\b',                 # aap kya karte
            r'\bkya\s+chal\s+raha\b',                     # kya chal raha hai
            r'\bsab\s+theek\b', r'\btheek\s+h(ai|o)\b', r'\bthik\s+h(ai|o)\b',
            r'\bdhanyavaad\b', r'\bshukriya\b', r'\bphir\s+milenge\b',
            r'\balvida\b', r'\bbye\s+bye\b',
            
            # Telugu / Tinglish
            r'\bela\s+unnav\b', r'\bela\s+unnaru\b',
            r'\bem\s+chesth?unnav\b', r'\bem\s+chesth?unnaru\b',
            r'\bem\s+chestunnavu\b', r'\bem\s+chesthunnar[u]?\b',
            r'\bbagunnara\b', r'\bbagunnava\b',
            r'\bbaagunnara\b', r'\bbaagunnava\b',
            r'\bnuvvu\s+ela\s+unnav\b', r'\bmeeru\s+ela\s+unnaru\b',
            r'\bdhanyavaadalu\b', r'\bvandanalu\b',
            r'\bemi\s+chesth?unnav\b', r'\bemi\s+chesth?unnaru\b',
            
            # Tamil / Tanglish
            r'\beppadi\s+irukk(a|inga|eenga)\b', r'\beppadi\s+irukkinga\b',
            r'\benna\s+pan[dn]?ra\b',                     # enna panra / enna pandra / enna pannra
            r'\benna\s+pand?reenga\b',                     # enna pandreenga / enna panreenga
            r'\benna\s+pannur(eeng|ing)[a]?\b',            # enna pannureenga / enna pannuring
            r'\benna\s+pannuva\b',                         # enna pannuva
            r'\benna\s+seyra\b', r'\benna\s+seyringa\b',  # enna seyra / enna seyringa
            r'\benna\s+seyyura\b',                         # enna seyyura
            r'\bnandri\b', r'\bpoitu\s+varen\b',
            r'\bnalla\s+irukk(a|inga|en)\b',
        ]
        
        for pattern in conversational_patterns:
            if re.search(pattern, q):
                return True
        
        # Very short queries with no KARE keywords are likely conversational
        words = q.split()
        if len(words) <= 2 and not self._has_kare_keywords(q):
            casual_short = {
                'ok', 'okay', 'yes', 'no', 'ya', 'nah', 'nope', 'yep',
                'thanks', 'thank', 'bye', 'hmm', 'hm', 'cool', 'nice',
                'great', 'awesome', 'fine', 'alright', 'sure', 'right',
                'haan', 'nahi', 'ji', 'aamaa', 'illa', 'avunu', 'ledu',
                'seri', 'haa',
            }
            if q in casual_short or (len(words) == 1 and words[0] in casual_short):
                return True
        
        return False

    def _has_kare_keywords(self, text: str) -> bool:
        """Check if text contains KARE University topic keywords"""
        kare_keywords = [
            'hostel', 'fee', 'fees', 'admission', 'course', 'placement', 'program',
            'department', 'faculty', 'bus', 'transport', 'mess', 'food', 'canteen',
            'scholarship', 'research', 'campus', 'facility', 'library', 'lab',
            'examination', 'exam', 'result', 'grade', 'cgpa', 'gpa',
            'btech', 'b.tech', 'mtech', 'm.tech', 'mba', 'mca', 'phd', 'doctorate',
            'engineering', 'degree', 'college', 'university', 'kare', 'kalasalingam',
            'apply', 'application', 'register', 'enroll', 'contact', 'phone', 'email',
            'website', 'address', 'location', 'principal', 'dean', 'director',
            'sports', 'gym', 'wifi', 'internet', 'medical', 'hospital',
            'cutoff', 'eligibility', 'criteria', 'intake', 'seat',
            # Hindi
            'hostel', 'fees', 'pravesh', 'vibhag', 'pariksha', 'naukri',
            # Telugu
            'hostel', 'fees', 'pravesham', 'vibhagam', 'pareeksha',
            # Tamil
            'viduthi', 'kattanam', 'padipu', 'thurai',
        ]
        return any(kw in text for kw in kare_keywords)

    # ------------------------------------------------------------------
    # Response generators for non-KARE queries
    # ------------------------------------------------------------------
    def _get_greeting_response(self, language: str) -> str:
        responses = {
            'en': "Hello! 👋 I'm KARE AI, your assistant for Kalasalingam Academy of Research and Education. I can help you with:\n\n• Admissions & Programs\n• Fees & Scholarships\n• Hostels & Mess\n• Placements & Facilities\n• Transport & Contact info\n\nWhat would you like to know?",
            'hi': "नमस्ते! 👋 मैं KARE AI हूँ, कलासलिंगम एकेडमी के लिए आपका सहायक। मैं इन विषयों में मदद कर सकता हूँ:\n\n• प्रवेश और कार्यक्रम\n• फीस और छात्रवृत्ति\n• हॉस्टल और मेस\n• प्लेसमेंट और सुविधाएं\n• परिवहन और संपर्क\n\nआप क्या जानना चाहते हैं?",
            'te': "నమస్కారం! 👋 నేను KARE AI, కలాసలింగం ఎకడమీ కొరకు మీ సహాయకుడను. నేను ఈ విషయాలలో సహాయం చేయగలను:\n\n• ప్రవేశాలు & కార్యక్రమాలు\n• ఫీజులు & స్కాలర్‌షిప్‌లు\n• హాస్టెళ్ళు & మెస్\n• ప్లేస్‌మెంట్లు & సౌకర్యాలు\n• రవాణా & సంప్రదింపు\n\nమీరు ఏమి తెలుసుకోవాలనుకుంటున్నారు?",
            'ta': "வணக்கம்! 👋 நான் KARE AI, கலாசலிங்கம் அகாடமிக்கான உங்கள் உதவியாளர். நான் இவற்றில் உதவ முடியும்:\n\n• சேர்க்கை & திட்டங்கள்\n• கட்டணம் & உதவித்தொகை\n• விடுதி & உணவகம்\n• வேலைவாய்ப்பு & வசதிகள்\n• போக்குவரத்து & தொடர்பு\n\nநீங்கள் என்ன தெரிந்துகொள்ள விரும்புகிறீர்கள்?",
            'kn': "ನಮಸ್ಕಾರ! 👋 ನಾನು KARE AI, ಕಲಾಸಲಿಂಗಮ್ ಅಕಾಡೆಮಿಗಾಗಿ ನಿಮ್ಮ ಸಹಾಯಕ. ನಾನು ಈ ವಿಷಯಗಳಲ್ಲಿ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ:\n\n• ಪ್ರವೇಶ & ಕಾರ್ಯಕ್ರಮಗಳು\n• ಶುಲ್ಕ & ವಿದ್ಯಾರ್ಥಿವೇತನ\n• ವಸತಿ & ಊಟ\n• ನಿಯೋಜನೆ & ಸೌಕರ್ಯ\n\nನೀವು ಏನು ತಿಳಿಯಲು ಬಯಸುತ್ತೀರಿ?",
            'ml': "നമസ്കാരം! 👋 ഞാൻ KARE AI, കലാസലിംഗം അക്കാദമിക്ക് വേണ്ടിയുള്ള നിങ്ങളുടെ സഹായി. ഞാൻ ഈ വിഷയങ്ങളിൽ സഹായിക്കാൻ കഴിയും:\n\n• പ്രവേശനം & പ്രോഗ്രാമുകൾ\n• ഫീസ് & സ്കോളർഷിപ്പ്\n• ഹോസ്റ്റൽ & ഭക്ഷണശാല\n• പ്ലെയ്സ്മെന്റ് & സൗകര്യങ്ങൾ\n\nനിങ്ങൾ എന്താണ് അറിയാൻ ആഗ്രഹിക്കുന്നത്?",
        }
        return responses.get(language, responses['en'])

    def _get_conversational_response(self, query: str, language: str) -> str:
        """Return appropriate conversational response"""
        q = query.lower().strip()
        
        # How are you
        how_are_you = {
            'en': "I'm doing great, thank you for asking! 😊 I'm KARE AI, ready to help you with information about Kalasalingam Academy. Ask me about admissions, fees, hostels, placements, or anything about KARE!",
            'hi': "मैं बहुत अच्छा हूँ, पूछने के लिए धन्यवाद! 😊 मैं KARE AI हूँ, कलासलिंगम एकेडमी के बारे में जानकारी देने के लिए तैयार। प्रवेश, फीस, हॉस्टल, प्लेसमेंट के बारे में पूछें!",
            'te': "నేను బాగున్నాను, అడిగినందుకు ధన్యవాదాలు! 😊 నేను KARE AI, కలాసలింగం ఎకడమీ గురించి సమాచారం ఇవ్వడానికి సిద్ధంగా ఉన్నాను. ప్రవేశాలు, ఫీజులు, హాస్టెల్, ప్లేస్‌మెంట్ల గురించి అడగండి!",
            'ta': "நான் நன்றாக இருக்கிறேன், கேட்டதற்கு நன்றி! 😊 நான் KARE AI, கலாசலிங்கம் அகாடமி பற்றிய தகவல்களை வழங்க தயாராக உள்ளேன். சேர்க்கை, கட்டணம், விடுதி, வேலைவாய்ப்பு பற்றி கேளுங்கள்!",
        }
        
        # Who are you / what are you
        who_are_you = {
            'en': "I'm KARE AI 🤖, an intelligent chatbot assistant for Kalasalingam Academy of Research and Education (KARE). I can help you find information about admissions, programs, fees, hostels, placements, facilities, transport, and more!",
            'hi': "मैं KARE AI 🤖 हूँ, कलासलिंगम एकेडमी ऑफ रिसर्च एंड एजुकेशन (KARE) के लिए एक बुद्धिमान चैटबॉट सहायक। मैं प्रवेश, कार्यक्रम, फीस, हॉस्टल, प्लेसमेंट और अधिक के बारे में जानकारी दे सकता हूँ!",
            'te': "నేను KARE AI 🤖, కలాసలింగం ఎకడమీ ఆఫ్ రీసెర్చ్ అండ్ ఎడ్యుకేషన్ (KARE) కొరకు తెలివైన చాట్‌బాట్ సహాయకుడిని. ప్రవేశాలు, కార్యక్రమాలు, ఫీజులు, హాస్టెళ్ళు, ప్లేస్‌మెంట్లు గురించి సమాచారం ఇవ్వగలను!",
            'ta': "நான் KARE AI 🤖, கலாசலிங்கம் அகாடமி ஆஃப் ரிசர்ச் அண்ட் எடுகேஷனுக்கான (KARE) அறிவார்ந்த சாட்பாட் உதவியாளர். சேர்க்கை, திட்டங்கள், கட்டணம், விடுதி, வேலைவாய்ப்பு பற்றிய தகவல்கள் வழங்க முடியும்!",
        }
        
        # Thank you
        thanks_resp = {
            'en': "You're welcome! 😊 Feel free to ask if you have more questions about KARE.",
            'hi': "आपका स्वागत है! 😊 KARE के बारे में और प्रश्न हों तो पूछें।",
            'te': "మీకు స్వాగతం! 😊 KARE గురించి మరిన్ని ప్రశ్నలు ఉంటే అడగండి.",
            'ta': "நன்றி! 😊 KARE பற்றி மேலும் கேள்விகள் இருந்தால் கேளுங்கள்.",
        }
        
        # Bye
        bye_resp = {
            'en': "Goodbye! 👋 Feel free to come back anytime you need help with KARE information.",
            'hi': "अलविदा! 👋 KARE के बारे में जानकारी चाहिए तो कभी भी आएं।",
            'te': "వీడ్కోలు! 👋 KARE సమాచారం కావాలంటే ఎప్పుడైనా రండి.",
            'ta': "பிரியாவிடை! 👋 KARE தகவல் தேவைப்பட்டால் எப்போது வேண்டுமானாலும் வாருங்கள்.",
        }
        
        # What are you doing
        what_doing = {
            'en': "I'm here to help you with information about Kalasalingam Academy (KARE)! 😊 You can ask me about admissions, fees, hostels, placements, programs, facilities, transport, and more!",
            'hi': "मैं यहाँ कलासलिंगम एकेडमी (KARE) की जानकारी देने के लिए हूँ! 😊 आप प्रवेश, फीस, हॉस्टल, प्लेसमेंट, कार्यक्रम, सुविधाओं के बारे में पूछ सकते हैं!",
            'te': "నేను కలాసలింగం ఎకడమీ (KARE) సమాచారంతో మీకు సహాయం చేయడానికి ఇక్కడ ఉన్నాను! 😊 ప్రవేశాలు, ఫీజులు, హాస్టెల్, ప్లేస్‌మెంట్లు, కార్యక్రమాలు గురించి అడగండి!",
            'ta': "நான் கலாசலிங்கம் அகாடமி (KARE) தகவல்களில் உங்களுக்கு உதவ இங்கே இருக்கிறேன்! 😊 சேர்க்கை, கட்டணம், விடுதி, வேலைவாய்ப்பு, திட்டங்கள் பற்றி கேளுங்கள்!",
        }
        
        # Route to appropriate response
        how_patterns = [
            r'how\s+are\s+you', r'how\s+do\s+you\s+do',
            r'kaise\s+h(o|ai)', r'kaisa\s+hai', r'kya\s+hal',
            r'ela\s+unnav', r'ela\s+unnaru', r'bagunnara', r'bagunnava',
            r'eppadi\s+irukk', r'nalla\s+irukk',
        ]
        who_patterns = [
            r'who\s+are\s+you', r'what\s+are\s+you', r'what\s+is\s+your\s+name',
            r"what'?s\s+your\s+name", r'what\s+can\s+you\s+do',
        ]
        what_doing_patterns = [
            r'what\s+(are\s+you|you)\s+doing', r'what\s+do\s+you\s+do',
            r'what\s+r\s+u\s+doing',
            r'kya\s+kart[aie]', r'kya\s+kar\s+rah[aei]', r'kya\s+chal\s+raha',
            r'tum\s+kya\s+kart[aie]', r'aap\s+kya\s+kart[eai]',
            r'em\s+chesth?unnav', r'em\s+chesth?unnaru', r'emi\s+chesth?unnav',
            r'enna\s+pan[dn]?ra', r'enna\s+pand?reenga', r'enna\s+pannur',
            r'enna\s+seyra', r'enna\s+seyringa', r'enna\s+seyyura',
            r'enna\s+pannuva',
        ]
        thanks_patterns = [r'thank', r'thanks', r'dhanyavaad', r'shukriya', r'nandri', r'dhanyavaadalu']
        bye_patterns = [r'\bbye\b', r'goodbye', r'see\s+you', r'alvida', r'phir\s+milenge', r'poitu\s+varen']
        
        # Check "what doing" BEFORE "who are you" (more specific first)
        for p in what_doing_patterns:
            if re.search(p, q):
                return what_doing.get(language, what_doing['en'])
        
        for p in how_patterns:
            if re.search(p, q):
                return how_are_you.get(language, how_are_you['en'])
        
        for p in who_patterns:
            if re.search(p, q):
                return who_are_you.get(language, who_are_you['en'])
        
        for p in thanks_patterns:
            if re.search(p, q):
                return thanks_resp.get(language, thanks_resp['en'])
        
        for p in bye_patterns:
            if re.search(p, q):
                return bye_resp.get(language, bye_resp['en'])
        
        # Generic conversational fallback
        fallback = {
            'en': "I'm KARE AI, here to help with information about Kalasalingam Academy. Try asking about admissions, fees, hostels, placements, programs, or facilities!",
            'hi': "मैं KARE AI हूँ, कलासलिंगम एकेडमी की जानकारी के लिए। प्रवेश, फीस, हॉस्टल, प्लेसमेंट, कार्यक्रम या सुविधाओं के बारे में पूछें!",
            'te': "నేను KARE AI, కలాసలింగం ఎకడమీ సమాచారం కోసం. ప్రవేశాలు, ఫీజులు, హాస్టెల్, ప్లేస్‌మెంట్లు, కార్యక్రమాలు గురించి అడగండి!",
            'ta': "நான் KARE AI, கலாசலிங்கம் அகாடமி தகவல்களுக்கு. சேர்க்கை, கட்டணம், விடுதி, வேலைவாய்ப்பு, திட்டங்கள் பற்றி கேளுங்கள்!",
        }
        return fallback.get(language, fallback['en'])

    def _get_no_info_response(self, query: str, language: str) -> str:
        """Response when no relevant information found"""
        responses = {
            'en': f"I couldn't find specific information about '{query}' in the KARE knowledge base. You can ask me about:\n\n• Admissions & Programs\n• Fees & Scholarships\n• Hostels & Mess\n• Placements & Facilities\n• Transport & Contact details\n\nPlease try rephrasing your question!",
            'hi': f"मुझे '{query}' के बारे में KARE डेटाबेस में जानकारी नहीं मिली। आप इनके बारे में पूछ सकते हैं:\n\n• प्रवेश और कार्यक्रम\n• फीस और छात्रवृत्ति\n• हॉस्टल और मेस\n• प्लेसमेंट और सुविधाएं\n\nकृपया अपना प्रश्न दोबारा पूछें!",
            'te': f"KARE డేటాబేస్‌లో '{query}' గురించి సమాచారం దొరకలేదు. మీరు వీటి గురించి అడగవచ్చు:\n\n• ప్రవేశాలు & కార్యక్రమాలు\n• ఫీజులు & స్కాలర్‌షిప్‌లు\n• హాస్టెళ్ళు & మెస్\n• ప్లేస్‌మెంట్లు & సౌకర్యాలు\n\nదయచేసి మీ ప్రశ్నను మార్చి అడగండి!",
            'ta': f"KARE தரவுத்தளத்தில் '{query}' பற்றிய தகவல் கிடைக்கவில்லை. நீங்கள் இவற்றைப் பற்றி கேட்கலாம்:\n\n• சேர்க்கை & திட்டங்கள்\n• கட்டணம் & உதவித்தொகை\n• விடுதி & உணவகம்\n• வேலைவாய்ப்பு & வசதிகள்\n\nதயவுசெய்து உங்கள் கேள்வியை மாற்றி கேளுங்கள்!",
            'kn': f"KARE ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ '{query}' ಬಗ್ಗೆ ಮಾಹಿತಿ ಸಿಗಲಿಲ್ಲ. ನೀವು ಪ್ರವೇಶ, ಶುಲ್ಕ, ವಸತಿ, ನಿಯೋಜನೆ ಬಗ್ಗೆ ಕೇಳಬಹುದು.",
            'ml': f"KARE ഡാറ്റാബേസിൽ '{query}' സംബന്ധിച്ച വിവരങ്ങൾ ലഭ്യമല്ല. പ്രവേശനം, ഫീസ്, ഹോസ്റ്റൽ, പ്ലെയ്സ്മെന്റ് എന്നിവയെക്കുറിച്ച് ചോദിക്കുക.",
        }
        return responses.get(language, responses['en'])

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------
    def _format_context(self, chunks: List[Dict]) -> str:
        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            source = chunk.get('source_file', 'unknown')
            text = chunk.get('text', '')
            score = chunk.get('similarity_score', 0)
            context_parts.append(f"[Source {idx}: {source} (relevance: {score:.2f})]\n{text}\n")
        return "\n".join(context_parts)

    # ------------------------------------------------------------------
    # LLM-based generation
    # ------------------------------------------------------------------
    def _generate_with_llm(self, query: str, context: str, language: str) -> str:
        try:
            logger.info("🤖 Generating response with Gemma2...")
            lang_instruction = ""
            if language != 'en':
                lang_map = {'ta': 'Tamil', 'te': 'Telugu', 'hi': 'Hindi', 'kn': 'Kannada', 'ml': 'Malayalam'}
                lang_name = lang_map.get(language, 'English')
                lang_instruction = f"\nIMPORTANT: Respond in {lang_name} language."
            
            response = self.llm.generate(
                prompt=query + lang_instruction,
                context=context,
                max_tokens=512,
                temperature=0.7
            )
            logger.info(f"✅ Generated {len(response)} chars")
            return response
        except Exception as e:
            logger.error(f"❌ LLM generation error: {e}")
            return self._generate_template_response(query, [], language)

    # ------------------------------------------------------------------
    # Template-based generation (no LLM)
    # ------------------------------------------------------------------
    def _generate_template_response(self, query: str, chunks: List[Dict], language: str) -> str:
        logger.info("📝 Using template response")
        if not chunks:
            return self._get_no_info_response(query, language)
        return self._format_natural_response(query, chunks, language)

    def _format_natural_response(self, query: str, chunks: List[Dict], language: str) -> str:
        """Format context into natural, well-organized response"""
        query_lower = query.lower()
        full_context = "\n".join([chunk.get('text', '') for chunk in chunks])
        
        # Parse structured data from context
        context_data = self._parse_markdown_context(full_context)
        
        # Try intent-specific formatting first
        specific = self._generate_intent_response(query, query_lower, context_data, full_context)
        if specific:
            return specific
        
        # Fallback: clean extraction from chunks
        return self._generate_clean_response(query, chunks, language)

    def _parse_markdown_context(self, context: str) -> dict:
        """Parse markdown context into structured data"""
        data = {
            'fees': [],
            'routes': [],
            'facilities': [],
            'programs': [],
            'general_info': [],
            'contacts': [],
        }
        
        lines = context.split('\n')
        current_item = {}
        
        for line in lines:
            line = line.strip()
            
            # Fee information
            if 'bedoccupancy:' in line.lower():
                beds = re.search(r'(\d+)', line)
                if beds:
                    current_item['beds'] = beds.group(1)
            if 'roomtype:' in line.lower():
                room_type = line.split(':', 1)[1].strip().replace('**', '').strip()
                current_item['room_type'] = room_type
            if any(x in line.lower() for x in ['ladieshostel:', 'menshostel:']):
                price = re.search(r'(\d+)', line)
                if price and 'nil' not in line.lower():
                    current_item['price'] = price.group(1)
                    if 'beds' in current_item and 'room_type' in current_item:
                        data['fees'].append(current_item.copy())
                        current_item = {}
            
            # Route information
            if 'destination:' in line.lower() or 'route:' in line.lower():
                route_info = line.split(':', 1)[1].strip().replace('**', '').strip()
                if route_info and len(route_info) > 2:
                    data['routes'].append(route_info)
            
            # Contact information
            if any(x in line.lower() for x in ['phone:', 'email:', 'mobile:', 'tel:']):
                data['contacts'].append(line.replace('- **', '').replace('**', '').strip())
            
            # General key-value information
            if line.startswith('- **') and ':' in line:
                key_value = line.replace('- **', '').replace('**', '')
                if ':' in key_value:
                    key, value = key_value.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key not in ['category', 'lastupdated', 'academicyear'] and value:
                        data['general_info'].append(f"{key.title()}: {value}")
        
        return data

    def _generate_intent_response(self, query: str, query_lower: str, data: dict, full_context: str) -> Optional[str]:
        """Generate response based on detected query intent"""
        
        # Document submission queries - COMPREHENSIVE
        if any(w in query_lower for w in ['document', 'submit', 'joining', 'required', 'need to bring', 'carry']):
            response = "**Documents Required for Joining** 📋\n\n"
            response += "**For UG (1st Year) Students:**\n"
            doc_count = 0
            mandatory_docs = {}
            
            # Extract all document requirements
            in_doc_section = False
            for line in full_context.split('\n'):
                line_lower = line.lower()
                
                # Start of document section
                if 'document' in line_lower and ('submission' in line_lower or 'requirement' in line_lower):
                    in_doc_section = True
                    continue
                
                # Extract document info
                if in_doc_section and ('mark sheet' in line_lower or 'transfer' in line_lower or 
                                       'conduct' in line_lower or 'certificate' in line_lower or
                                       'aadhar' in line_lower or 'photograph' in line_lower):
                    cleaned = line.replace('**', '').replace('- ', '').strip()
                    if cleaned and ':' in cleaned:
                        key = cleaned.split(':')[0].strip()
                        val = cleaned.split(':', 1)[1].strip()
                        if val and val.lower() not in ['nil', 'n/a', '']:
                            mandatory_docs[key] = val
                            doc_count += 1
            
            # Prepare comprehensive default document list
            comprehensive_docs = {
                "10th Mark Sheet": "Original + 3 Xerox copies",
                "12th Mark Sheet": "Original + 3 Xerox copies",
                "Transfer Certificate (TC)": "Original + 1 Xerox copy",
                "Conduct Certificate": "Original + 1 Xerox copy",
                "Medical Fitness Certificate": "Original (from registered medical practitioner)",
                "Community Certificate": "1 Xerox copy (if applicable - BC/MBC/SC/ST)",
                "Passport Size Photographs": "4-6 copies (3x4 cm, recent and transparent background)",
                "Aadhar Card": "Copy",
                "Entrance Exam Scorecard": "If applicable (JEE/GATE/CAT/etc.)",
                "Migration Certificate": "If applicable (for inter-state transfers)",
                "Income Certificate": "For scholarship eligibility verification",
            }
            
            # Merge extracted docs with comprehensive list (extracted takes priority where available)
            for key in comprehensive_docs:
                if key not in mandatory_docs:
                    mandatory_docs[key] = comprehensive_docs[key]
            
            # Display all documents clearly
            for i, (doc_type, details) in enumerate(list(mandatory_docs.items())[:15], 1):
                response += f"{i}. **{doc_type}**: {details}\n"
            response += f"\n✅ **Total Required Documents**: {len(mandatory_docs)}\n"
            
            # Add submission info from context
            if 'at the time of' in full_context.lower() and 'joining' in full_context.lower():
                response += "\n⏰ **When to Submit**: At the time of joining college (during first week)\n"
            if 'admissions office' in full_context.lower():
                response += "📍 **Where to Submit**: Admissions Office\n"
            
            # Add important instructions
            response += "\n**Important Instructions:**\n"
            response += "• Original documents required for verification\n"
            response += "• Photocopy (Xerox) certificates must be clear and legible\n"
            response += "• Medical fitness certificate from registered medical practitioner\n"
            response += "• Community certificate only if applicable (BC/MBC/SC/ST)\n"
            response += "• Entrance exam scorecard if applicable\n"
            response += "\n📞 **Contact**: admissions@klu.ac.in | +91 73977 60760"
            return response
        
        # Eligibility criteria queries - COMPREHENSIVE
        if any(w in query_lower for w in ['eligibility', 'criteria', 'qualification', 'eligible', 'requirement', 'needed']):
            response = "**Eligibility Criteria** 🎓\n\n"
            
            # Extract all program eligibility from context
            eligibility_info = {}
            current_prog = None
            
            for line in full_context.split('\n'):
                line_clean = line.replace('**', '').strip()
                
                # Detect program headers
                if any(p in line_clean.upper() for p in ['B.TECH', 'B.SC', 'BBA', 'BCA', 'M.TECH', 'MBA', 'M.SC', 'MCA', 'PHD']):
                    prog_match = re.search(r'(B\.?TECH|B\.?SC|BBA|BCA|M\.?TECH|MBA|M\.?SC|MCA|PHD)', line_clean, re.IGNORECASE)
                    if prog_match:
                        current_prog = prog_match.group(1).upper()
                        eligibility_info[current_prog] = {'qual': '', 'marks': ''}
                
                # Extract qualification and marks requirements
                if current_prog and 'qualification' in line.lower():
                    qual_match = re.search(r'Qualification[:\s]+([^*]*)', line, re.IGNORECASE)
                    if qual_match:
                        eligibility_info[current_prog]['qual'] = qual_match.group(1).strip()
                
                if current_prog and ('minimum' in line.lower() or 'aggregate' in line.lower()):
                    marks_match = re.search(r'([0-9]+%)', line)
                    if marks_match:
                        eligibility_info[current_prog]['marks'] = marks_match.group(1).strip()
            
            # Display eligibility info - always show comprehensive list
            response += "**UNDERGRADUATE (UG) PROGRAMS:**\n\n"
            response += "**B.Tech:**\n"
            response += "  • Qualification: 10+2 with PCM (Physics, Chemistry, Mathematics)\n"
            response += "  • Marks: 50% aggregate (45% for reserved categories)\n"
            response += "  • Entrance Exams: JEE Main, TNEA, KARE Entrance Test\n\n"
            
            response += "**B.Sc:**\n"
            response += "  • Qualification: 10+2 with relevant subjects\n"
            response += "  • Marks: 50% aggregate\n\n"
            
            response += "**BBA:**\n"
            response += "  • Qualification: 10+2 in any stream\n"
            response += "  • Marks: 50% aggregate\n\n"
            
            response += "**BCA:**\n"
            response += "  • Qualification: 10+2 with Mathematics\n"
            response += "  • Marks: 50% aggregate\n\n"
            
            response += "**POSTGRADUATE (PG) PROGRAMS:**\n\n"
            response += "**M.Tech / M.E:**\n"
            response += "  • Qualification: B.Tech/B.E in relevant branch\n"
            response += "  • Marks: 55% aggregate (50% for reserved)\n"
            response += "  • Entrance Exams: GATE (preferred), KARE Entrance Test\n\n"
            
            response += "**MBA:**\n"
            response += "  • Qualification: Any bachelor's degree\n"
            response += "  • Marks: 50% aggregate\n"
            response += "  • Entrance Exams: CAT, MAT, TANCET, KARE Entrance Test\n\n"
            
            response += "**M.Sc:**\n"
            response += "  • Qualification: B.Sc in relevant subject\n"
            response += "  • Marks: 55% aggregate\n\n"
            
            response += "**M.C.A.:**\n"
            response += "  • Qualification: Bachelor's degree with Mathematics\n"
            response += "  • Marks: 50% aggregate\n"
            response += "  • Entrance Exams: TANCET, KARE Entrance Test\n\n"
            
            response += "**PhD:**\n"
            response += "  • Qualification: Master's degree in relevant field\n"
            response += "  • Marks: 55% aggregate\n"
            response += "  • Entrance Exams: NET, GATE, KARE Research Entrance Test\n"
            
            response += "\n**IMPORTANT NOTES:**\n"
            response += "✓ Admissions are purely merit-based\n"
            response += "✓ Eligibility criteria may change annually - check official website\n"
            response += "✓ Early application recommended\n"
            response += "✓ Document verification is mandatory\n"
            response += "✓ Incomplete applications will be rejected\n"
            
            response += f"\n🔗 **Visit**: www.klu.ac.in/admissions\n"
            response += f"📧 **Email**: admissions@klu.ac.in\n"
            response += f"📞 **Contact**: +91 73977 60760"
            return response
        
        # Fresher hostel allocation queries
        if any(w in query_lower for w in ['fresher', 'freshman', 'first year', '1st year', 'newly admitted', 'new student']):
            if 'fresher' in full_context.lower() or 'mh1' in full_context.lower():
                response = "**Hostel Allocation for Freshers** 🏨\n\n"
                has_info = False
                # Extract fresher-specific hostel info
                for line in full_context.split('\n'):
                    line_lower = line.lower()
                    if any(w in line_lower for w in ['fresher', 'mh1', 'mandela', 'allocation', '1st year', 'newly']):
                        cleaned = line.replace('**', '').replace('- ', '').strip()
                        if cleaned and len(cleaned) > 5:
                            response += f"• {cleaned}\n"
                            has_info = True
                if has_info:
                    response += "\n📍 **Common fresher hostels:** Nelson Mandela (MH1) and MH5\n"
                    response += "📋 Allocation is done on **first-come-first-serve basis**\n"
                    response += "💡 Prefer 5-bed or 4-bed sharing for better costs"
                    return response
        
        # Hostel fee queries
        if 'hostel' in query_lower and any(w in query_lower for w in ['fee', 'cost', 'price', 'charge', 'kitna', 'entha', 'evvalavu']):
            response = "**Hostel Fees 2025-2026** 💰\n\n"
            has_fees = False
            # Extract actual fee data from chunks
            for line in full_context.split('\n'):
                line_lower = line.lower()
                if any(kw in line_lower for kw in ['sharing:', 'occupancy:', 'fee:', 'ladies', 'men']):
                    cleaned = line.replace('**', '').replace('- ', '').strip()
                    if cleaned and ':' in cleaned and len(cleaned) > 10:
                        response += f"• {cleaned}\n"
                        has_fees = True
            if has_fees:
                response += "\n✅ **Mess charges included** in hostel fees\n"
                response += "📞 Contact: hostel@klu.ac.in | +91 4563 289 070"
                return response
        
        # Transport / bus fare queries - IMPROVED to show actual fares
        if any(w in query_lower for w in ['bus', 'transport', 'route', 'fare', 'ticket', 'cost', 'price']):
            response = "**Bus Transport Information** 🚌\n\n"
            has_info = False
            
            # Extract both routes AND fares/costs
            for line in full_context.split('\n'):
                line_lower = line.lower()
                # Look for fare/cost information
                if any(kw in line_lower for kw in ['farerange:', 'annualfare:', 'totalfare:', 'cost:', 'price:', '₹', '₹']):
                    cleaned = line.replace('**', '').replace('- ', '').strip()
                    if cleaned and len(cleaned) > 5:
                        response += f"• {cleaned}\n"
                        has_info = True
                        break
            
            if has_info:
                # Add specific routes if available
                response += "\n**Available Routes:**\n"
                route_count = 0
                for line in full_context.split('\n'):
                    if 'route:' in line.lower() and route_count < 5:
                        cleaned = line.replace('**', '').replace('- ', '').strip()
                        if cleaned and ':' in cleaned:
                            response += f"• {cleaned.split(':')[1].strip()}\n"
                            route_count += 1
                response += "\n📞 For booking: Contact Transport Office\n"
                response += "📍 Pickup points on campus"
                return response
        
        # Program / course queries - IMPROVED to show all programs
        if any(w in query_lower for w in ['program', 'course', 'degree', 'btech', 'mtech', 'mba', 'mca', 'phd', 'branch', 'stream']):
            programs_found = {}
            for line in full_context.split('\n'):
                line_clean = line.replace('**', '').strip()
                # Look for program names (usually in Name: fields)
                if ' - **Code:' in line or ' - **code:' in line or line.startswith('- **Name:'):
                    prog_info = re.search(r'Name:\s*([^-*]+)', line, re.IGNORECASE)
                    if prog_info:
                        prog_name = prog_info.group(1).strip()
                        if prog_name and len(prog_name) > 3:
                            programs_found[prog_name.lower()] = prog_name
                # Also catch plain program mentions in BIG CAPS or key patterns
                if any(p in line.upper() for p in ['B.TECH', 'M.TECH', 'MBA', 'MCA', 'M.SC', 'M.PHIL', 'PH.D']):
                    if '**' in line or ':' in line:
                        cleaned = line.replace('**', '').replace('- ', '').strip()
                        if cleaned and len(cleaned) > 5:
                            programs_found[cleaned.lower()] = cleaned
            
            if programs_found:
                response = "**Programs Offered at KARE** 🎓\n\n"
                for prog in list(programs_found.values())[:12]:
                    response += f"• {prog}\n"
                response += "\n📋 For specializations, duration, and admission criteria:\n"
                response += "🔗 Visit: www.klu.ac.in/admissions\n"
                response += "📞 Contact: admissions@klu.ac.in | +91 73977 60760"
                return response
        
        # Contact queries
        if any(w in query_lower for w in ['contact', 'phone', 'email', 'number', 'call', 'reach', 'address']):
            response = "**Contact Information** 📞\n\n"
            contacts = {}
            for line in full_context.split('\n'):
                line_lower = line.lower()
                if 'email:' in line_lower or 'phone:' in line_lower:
                    cleaned = line.replace('**', '').replace('- ', '').strip()
                    if cleaned and ':' in cleaned:
                        key = cleaned.split(':')[0].strip()
                        val = cleaned.split(':', 1)[1].strip()
                        if val and val.lower() not in ['', 'nil', 'n/a']:
                            contact_key = f"{key} ({val[:30]})"
                            contacts[contact_key] = True
            
            if contacts:
                for contact in list(contacts.keys())[:8]:
                    response += f"• {contact}\n"
                return response
        
        return None  # No specific intent matched

    def _generate_clean_response(self, query: str, chunks: List[Dict], language: str) -> str:
        """Generate a clean, readable response from chunks"""
        # Always use English acknowledgment here — Gemini will translate the whole
        # response in Step 5 if the target language is not English.
        ack = "Here's what I found:"
        
        # Collect all meaningful bullet points from all chunks
        all_bullets = []
        seen_bullets = set()
        
        for chunk in chunks:
            text = chunk.get('text', '').strip()
            if not text:
                continue
            
            # Extract clean bullets from each chunk
            bullets = self._extract_clean_bullets(text)
            for bullet in bullets:
                # Deduplicate
                key = bullet.lower().strip()[:80]
                if key not in seen_bullets and len(bullet.strip()) > 3:
                    seen_bullets.add(key)
                    all_bullets.append(bullet)
        
        if not all_bullets:
            return self._get_no_info_response(query, language)
        
        # Limit to top 12 bullets for readability
        all_bullets = all_bullets[:12]
        
        response = ack + "\n\n"
        response += "\n".join(f"• {b}" for b in all_bullets)
        
        # Add source attribution
        source = chunks[0].get('source_file', 'KARE')
        response += f"\n\n*(Source: {source})*"
        
        return response

    def _extract_clean_bullets(self, text: str) -> list:
        """Extract clean, readable bullet points from raw markdown text"""
        bullets = []
        lines = text.split('\n')
        
        skip_keys = {
            'category', 'lastupdated', 'last_updated', 'academicyear',
            'academic_year', 'source', 'type', 'id', 'index',
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip markdown headers (##, ###) - extract as section titles
            if line.startswith('#'):
                title = line.lstrip('#').strip()
                if title and len(title) > 3 and not any(sk in title.lower() for sk in skip_keys):
                    bullets.append(f"**{title}**")
                continue
            
            # Skip "Item N:" prefixes
            line = re.sub(r'^Item\s+\d+\s*:\s*', '', line).strip()
            
            # Handle "- **Key:** Value" markdown pattern
            m = re.match(r'^-\s*\*?\*?([^:*]+)\*?\*?\s*:\s*(.+)$', line)
            if m:
                key = m.group(1).strip().lower()
                value = m.group(2).strip().rstrip('*').strip()
                # Skip metadata fields
                if key in skip_keys:
                    continue
                if value and value.lower() not in ['nil', 'n/a', '-', 'none', '']:
                    bullets.append(f"{key.title()}: {value}")
                continue
            
            # Handle "- text" plain bullets
            if line.startswith('- '):
                clean = line[2:].replace('**', '').strip()
                if clean and len(clean) > 3:
                    # Check if it's a key:value pair
                    if ':' in clean:
                        key_part = clean.split(':', 1)[0].strip().lower()
                        if key_part in skip_keys:
                            continue
                    bullets.append(clean)
                continue
            
            # Handle "Key: Value" pairs (no dash prefix)
            if ':' in line and not line.startswith('http'):
                key_part = line.split(':', 1)[0].strip().lower()
                value_part = line.split(':', 1)[1].strip()
                if key_part in skip_keys:
                    continue
                if value_part and len(value_part) > 1:
                    clean_line = line.replace('**', '').strip()
                    bullets.append(clean_line)
                continue
            
            # Handle pipe-separated data
            if '|' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                for part in parts:
                    if ':' in part:
                        k = part.split(':', 1)[0].strip().lower()
                        if k not in skip_keys:
                            bullets.append(part.replace('**', '').strip())
                    elif len(part) > 3:
                        bullets.append(part.replace('**', '').strip())
                continue
            
            # Plain text (only if substantial)
            clean = line.replace('**', '').replace('*', '').strip()
            if len(clean) > 15:
                bullets.append(clean)
        
        return bullets

    def _clean_text(self, text: str) -> str:
        """Clean and format text for readability"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Format pipe-separated key-value pairs
        if '|' in text:
            parts = text.split('|')
            formatted = []
            for part in parts:
                part = part.strip()
                if ':' in part:
                    key, val = part.split(':', 1)
                    formatted.append(f"• {key.strip()}: {val.strip()}")
                elif part:
                    formatted.append(f"• {part}")
            return '\n'.join(formatted)
        
        return text

    # ------------------------------------------------------------------
    # Utility response helpers
    # ------------------------------------------------------------------
    def _get_acknowledgment(self, language: str) -> str:
        ack = {
            'en': "Here's what I found:",
            'hi': "यहाँ मुझे क्या मिला:",
            'te': "ఇక్కడ నేను కనుగొన్నవి:",
            'ta': "நான் கண்டது இதோ:",
            'kn': "ನಾನು ಕಂಡುಕೊಂಡದ್ದು ಇಲ್ಲಿದೆ:",
            'ml': "ഞാൻ കണ്ടെത്തിയത് ഇതാണ്:",
        }
        return ack.get(language, ack['en'])

    def _get_closing(self, language: str) -> str:
        closing = {
            'en': "Is there anything else you'd like to know?",
            'hi': "क्या आप और कुछ जानना चाहते हैं?",
            'te': "మీరు ఇంకా ఏదైనా తెలుసుకోవాలనుకుంటున్నారా?",
            'ta': "வேறு ஏதாவது தெரிந்து கொள்ள விரும்புகிறீர்களா?",
            'kn': "ನೀವು ಇನ್ನೇನಾದರೂ ತಿಳಿಯಲು ಬಯಸುತ್ತೀರಾ?",
            'ml': "നിങ്ങൾക്ക് മറ്റെന്തെങ്കിലും അറിയണോ?",
        }
        return closing.get(language, closing['en'])

    def get_system_info(self) -> Dict:
        return {
            "vector_store_loaded": self.vector_store.index is not None,
            "total_chunks": len(self.vector_store.chunks) if self.vector_store.chunks else 0,
            "llm_available": self.llm_available,
            "llm_model": self.llm.get_model_info() if self.llm else None,
        }


# Global instance
_rag_generator = None

def get_rag_generator(
    model_path: Optional[str] = None,
    use_llm: bool = False,  # Default to False until model is downloaded
    **kwargs
) -> RAGResponseGenerator:
    """Get or create global RAG generator instance"""
    global _rag_generator
    
    if _rag_generator is None:
        _rag_generator = RAGResponseGenerator(
            model_path=model_path,
            use_llm=use_llm,
            **kwargs
        )
    
    return _rag_generator


def generate_rag_response(query: str, language: str = 'en', top_k: int = 3) -> str:
    """Utility function for generating RAG responses"""
    generator = get_rag_generator()
    return generator.generate_response(query, language, top_k)


if __name__ == "__main__":
    # Test RAG system
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing RAG Response Generator")
    print("=" * 70)
    
    # Initialize (without LLM for now)
    rag = RAGResponseGenerator(use_llm=False)
    
    print(f"\n📊 System Info:")
    info = rag.get_system_info()
    print(f"   Vector Store Loaded: {info['vector_store_loaded']}")
    print(f"   Total Chunks: {info['total_chunks']}")
    print(f"   LLM Available: {info['llm_available']}")
    
    # Test queries
    test_queries = [
        ("What is the hostel fee?", "en"),
        ("Tell me about bus routes", "en"),
        ("admission process kya hai?", "hi"),
        ("programs gurinchi cheppandi", "te"),
    ]
    
    for query, lang in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query} (Language: {lang})")
        print(f"{'='*70}")
        
        response = rag.generate_response(query, language=lang, top_k=2)
        print(f"\nResponse:\n{response}")
