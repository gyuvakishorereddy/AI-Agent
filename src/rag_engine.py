"""
RAG Response Generator with FAISS Vector Store + Gemma2-9B Q4
Integrates vector similarity search with LLM generation for intelligent responses
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from vector_store import VectorStoreManager
from gemma2_llm import Gemma2LLM

logger = logging.getLogger(__name__)


class RAGResponseGenerator:
    """Generate responses using RAG (Retrieval-Augmented Generation) with FAISS + Gemma2"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vector_store_path: str = "faiss_index",
        data_dir: str = "data",
        use_llm: bool = True
    ):
        """
        Initialize RAG system
        
        Args:
            model_path: Path to Gemma2-9B Q4 model file (optional, will search automatically)
            vector_store_path: Path to FAISS index directory
            data_dir: Path to JSON data files
            use_llm: Whether to use Gemma2 LLM (if False, uses template responses)
        """
        self.use_llm = use_llm
        self.llm_available = False
        
        # Initialize vector store
        logger.info("🔄 Initializing RAG system...")
        
        self.vector_store = VectorStoreManager(
            data_dir=data_dir,
            vector_store_path=vector_store_path
        )
        
        # Load existing vector store
        if not self.vector_store.load_vector_store():
            logger.warning("⚠️ Vector store not found. Please run build_vector_store.py first.")
            logger.info("   Run: python build_vector_store.py")
        
        # Initialize LLM (if requested)
        self.llm = None
        if use_llm:
            try:
                logger.info("🤖 Initializing Gemma2-9B Q4 LLM...")
                self.llm = Gemma2LLM(
                    model_path=model_path,
                    n_ctx=4096,
                    n_gpu_layers=0,  # CPU only (set higher for GPU)
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
                logger.info("   Falling back to template-based responses")
                self.llm_available = False
        else:
            logger.info("✅ RAG system ready (Vector Store only - template mode)")
    
    def generate_response(self, query: str, language: str = 'en', top_k: int = 3) -> str:
        """
        Generate response using RAG pipeline
        
        Args:
            query: User query
            language: Response language (en, ta, te, hi, kn, ml)
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Generated response
        """
        logger.info(f"🎯 Query: {query[:50]}... (Language: {language})")
        
        # Check for greeting queries
        if self._is_greeting(query):
            return self._get_greeting_response(language)
        
        # Step 1: Retrieve relevant context from vector store
        context_chunks = self.vector_store.search(query, top_k=top_k)
        
        if not context_chunks:
            logger.warning("⚠️ No relevant information found")
            return self._get_no_info_response(query, language)
        
        # Step 2: Format context
        context_text = self._format_context(context_chunks)
        logger.info(f"📚 Retrieved {len(context_chunks)} relevant chunks")
        
        # Step 3: Generate response
        if self.llm_available and self.llm:
            # Use Gemma2 LLM for generation
            response = self._generate_with_llm(query, context_text, language)
        else:
            # Use template-based response
            response = self._generate_template_response(query, context_chunks, language)
        
        return response
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string"""
        context_parts = []
        
        for idx, chunk in enumerate(chunks, 1):
            source = chunk.get('source_file', 'unknown')
            text = chunk.get('text', '')
            score = chunk.get('similarity_score', 0)
            
            context_parts.append(f"[Source {idx}: {source} (relevance: {score:.2f})]\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _generate_with_llm(self, query: str, context: str, language: str) -> str:
        """Generate response using Gemma2 LLM"""
        try:
            logger.info("🤖 Generating response with Gemma2...")
            
            # Add language instruction if not English
            lang_instruction = ""
            if language != 'en':
                lang_map = {
                    'ta': 'Tamil',
                    'te': 'Telugu',
                    'hi': 'Hindi',
                    'kn': 'Kannada',
                    'ml': 'Malayalam'
                }
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
    
    def _generate_template_response(self, query: str, chunks: List[Dict], language: str) -> str:
        """Generate template-based response (fallback)"""
        logger.info("📝 Using template response")
        
        if not chunks:
            return self._get_no_info_response(query, language)
        
        # Use smart extraction instead of raw dump
        response = self._format_natural_response(query, chunks)
        
        return response
    
    def _format_natural_response(self, query: str, chunks: List[Dict]) -> str:
        """Format context into natural, conversational response using LLM-style generation"""
        import re
        
        query_lower = query.lower()
        
        # Combine all chunk text
        full_context = "\n".join([chunk.get('text', '') for chunk in chunks])
        
        # Use a simple LLM-like prompt system to generate natural responses
        system_prompt = """You are KARE AI Assistant. Given the context from university knowledge base, answer the user's question in a clear, natural, and conversational way. 

Extract relevant information from the context and present it in a well-organized format.
For fees, show amounts clearly with currency symbols.
Be concise but complete.
Do not repeat metadata or technical formatting."""

        # Parse the context to extract structured information
        context_data = self._parse_markdown_context(full_context)
        
        # Generate response based on query type
        response = self._generate_llm_style_response(query, query_lower, context_data, full_context)
        
        return response
    
    def _parse_markdown_context(self, context: str) -> dict:
        """Parse markdown context into structured data"""
        import re
        
        data = {
            'fees': [],
            'routes': [],
            'facilities': [],
            'programs': [],
            'general_info': []
        }
        
        lines = context.split('\n')
        current_item = {}
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Extract fee information
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
                    
                    # Complete the fee item
                    if 'beds' in current_item and 'room_type' in current_item:
                        data['fees'].append(current_item.copy())
                        current_item = {}
            
            # Extract route information
            if 'destination:' in line.lower() or 'route:' in line.lower():
                route_info = line.split(':', 1)[1].strip().replace('**', '').strip()
                if route_info and len(route_info) > 2:
                    data['routes'].append(route_info)
            
            # Extract general information
            if line.startswith('- **') and ':' in line:
                key_value = line.replace('- **', '').replace('**', '')
                if ':' in key_value:
                    key, value = key_value.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key not in ['category', 'lastupdated', 'academicyear'] and value:
                        data['general_info'].append(f"{key.title()}: {value}")
        
        return data
    
    def _generate_llm_style_response(self, query: str, query_lower: str, data: dict, full_context: str) -> str:
        """Generate natural language response like an LLM would"""
        
        # Hostel fee queries
        if 'hostel' in query_lower and any(word in query_lower for word in ['fee', 'cost', 'price', 'charge']):
            if data['fees']:
                response = "Here are the hostel fees for 2025-2026 academic year:\n\n"
                
                # Group by bed occupancy
                for fee in data['fees']:
                    beds = fee.get('beds', '')
                    room_type = fee.get('room_type', '')
                    price = fee.get('price', '')
                    
                    if beds and room_type and price:
                        response += f"• {beds}-bed sharing ({room_type}): ₹{price:,} per year\n"
                
                response += "\n💡 Note: Mess fees are included in the hostel fees."
                return response
        
        # Transport queries
        if any(word in query_lower for word in ['bus', 'transport', 'route', 'fare']):
            if data['routes']:
                response = "Here are the available bus routes:\n\n"
                for route in data['routes'][:10]:
                    response += f"• {route}\n"
                response += "\nFor detailed timings and fares, please contact the transport office."
                return response
        
        # Program queries
        if any(word in query_lower for word in ['program', 'course', 'degree', 'btech', 'mtech']):
            programs = []
            for line in full_context.split('\n'):
                if any(word in line.lower() for word in ['btech', 'mtech', 'engineering', 'program', 'degree']):
                    cleaned = line.replace('**', '').replace('- ', '').replace('#', '').strip()
                    if cleaned and 5 < len(cleaned) < 100:
                        programs.append(cleaned)
            
            if programs:
                response = "KARE offers the following programs:\n\n"
                for prog in programs[:15]:
                    response += f"• {prog}\n"
                return response
        
        # General hostel info
        if 'hostel' in query_lower:
            if data['general_info']:
                response = "Hostel Information:\n\n"
                for info in data['general_info'][:10]:
                    if any(word in info.lower() for word in ['hostel', 'mens', 'ladies', 'separate']):
                        response += f"• {info}\n"
                return response
        
        # Default: Show relevant general information
        if data['general_info']:
            response = "Here's what I found:\n\n"
            for info in data['general_info'][:8]:
                response += f"• {info}\n"
            return response
        
        # Final fallback
        return "I found relevant information in the knowledge base. Please try asking more specifically about admissions, fees, hostels, placements, programs, or facilities."
    
    def _clean_text(self, text: str) -> str:
        """Clean and format text for readability"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Format key-value pairs nicely
        if '|' in text:
            parts = text.split('|')
            formatted = []
            for part in parts:
                part = part.strip()
                if ':' in part:
                    key, val = part.split(':', 1)
                    formatted.append(f"• {key.strip()}: {val.strip()}")
                else:
                    formatted.append(f"• {part}")
            return '\n'.join(formatted)
        
        return text
    
    def _is_greeting(self, query: str) -> bool:
        """Check if query is a greeting"""
        greetings = [
            'hello', 'hi', 'hey', 'namaste', 'vanakkam',
            'how are you', 'how do you do', 'whats up'
        ]
        query_lower = query.lower().strip()
        
        # Very short greeting-like queries
        if len(query_lower) < 20:
            for greeting in greetings:
                if greeting in query_lower:
                    # Make sure it's not about university topics
                    uni_keywords = ['hostel', 'fee', 'admission', 'course', 'placement']
                    if not any(kw in query_lower for kw in uni_keywords):
                        return True
        
        return False
    
    def _get_greeting_response(self, language: str) -> str:
        """Get greeting response"""
        responses = {
            'en': "Hello! I'm KARE AI, your intelligent assistant for Kalasalingam Academy of Research and Education. I can help you with information about admissions, programs, fees, hostels, placements, facilities, and more. What would you like to know?",
            
            'hi': "नमस्ते! मैं KARE AI हूँ, कलासलिंगम एकेडमी के लिए आपका बुद्धिमान सहायक। मैं प्रवेश, कार्यक्रम, फीस, हॉस्टल, प्लेसमेंट, सुविधाओं और अधिक के बारे में जानकारी में मदद कर सकता हूँ। आप क्या जानना चाहते हैं?",
            
            'te': "నమస్కారం! నేను KARE AI, కలాసలింగం ఎకడమీ కొరకు మీ తెలివైన సహాయకుడను. నేను ప్రవేశాలు, కార్యక్రమాలు, ఫీజు, హాస్టెల్, ప్లేస్‌మెంట్స్, సౌకర్యాలు మరియు మరిన్ని గురించి సమాచారంతో సహాయం చేయగలను. మీరు ఏమి తెలుసుకోవాలనుకుంటున్నారు?",
            
            'ta': "வணக்கம்! நான் KARE AI, கலாசலிங்கம் அகாடமிக்கான உங்கள் அறிவார்ந்த உதவியாளர். சேர்க்கை, திட்டங்கள், கட்டணம், விடுதி, வேலைவாய்ப்பு, வசதிகள் மற்றும் பலவற்றைப் பற்றிய தகவல்களில் நான் உங்களுக்கு உதவ முடியும். நீங்கள் என்ன தெரிந்துகொள்ள விரும்புகிறீர்கள்?",
            
            'kn': "ನಮಸ್ಕಾರ! ನಾನು KARE AI, ಕಲಾಸಲಿಂಗಮ್ ಅಕಾಡೆಮಿಗಾಗಿ ನಿಮ್ಮ ಬುದ್ಧಿವಂತ ಸಹಾಯಕ. ಪ್ರವೇಶಗಳು, ಕಾರ್ಯಕ್ರಮಗಳು, ಶುಲ್ಕ, ವಸತಿ, ನಿಯೋಜನೆಗಳು, ಸೌಕರ್ಯಗಳು ಮತ್ತು ಹೆಚ್ಚಿನದರ ಬಗ್ಗೆ ಮಾಹಿತಿಯೊಂದಿಗೆ ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ. ನೀವು ಏನು ತಿಳಿಯಲು ಬಯಸುತ್ತೀರಿ?",
            
            'ml': "നമസ്കാരം! ഞാൻ KARE AI, കലാസലിംഗം അക്കാദമിക്ക് വേണ്ടിയുള്ള നിങ്ങളുടെ ബുദ്ധിപരമായ സഹായി. പ്രവേശനം, പ്രോഗ്രാമുകൾ, ഫീസ്, ഹോസ്റ്റൽ, പ്ലെയ്സ്മെന്റുകൾ, സൗകര്യങ്ങൾ എന്നിവയെക്കുറിച്ചുള്ള വിവരങ്ങളിൽ ഞാൻ നിങ്ങളെ സഹായിക്കാൻ കഴിയും. നിങ്ങൾ എന്താണ് അറിയാൻ ആഗ്രഹിക്കുന്നത്?",
        }
        
        return responses.get(language, responses['en'])
    
    def _get_no_info_response(self, query: str, language: str) -> str:
        """Response when no information found"""
        responses = {
            'en': f"I don't have specific information about '{query}'. Please try asking about admissions, fees, hostels, placements, programs, facilities, or contact information.",
            
            'hi': f"मुझे '{query}' के बारे में विशिष्ट जानकारी नहीं है। कृपया प्रवेश, फीस, हॉस्टल, प्लेसमेंट, कार्यक्रम, सुविधाएं या संपर्क जानकारी के बारे में पूछें।",
            
            'te': f"నాకు '{query}' గురించి నిర్దిష్ట సమాచారం లేదు. దయచేసి ప్రవేశాలు, ఫీజు, హాస్టెల్, ప్లేస్‌మెంట్స్, కార్యక్రమాలు, సౌకర్యాలు లేదా సంప్రదింపు సమాచారం గురించి అడగండి।",
            
            'ta': f"எனக்கு '{query}' பற்றிய குறிப்பிட்ட தகவல் இல்லை. சேர்க்கை, கட்டணம், விடுதி, வேலைவாய்ப்பு, திட்டங்கள், வசதிகள் அல்லது தொடர்பு தகவல் பற்றி கேட்கவும்.",
            
            'kn': f"ನನಗೆ '{query}' ಬಗ್ಗೆ ನಿರ್ದಿಷ್ಟ ಮಾಹಿತಿ ಇಲ್ಲ. ದಯವಿಟ್ಟು ಪ್ರವೇಶಗಳು, ಶುಲ್ಕ, ವಸತಿ, ನಿಯೋಜನೆಗಳು, ಕಾರ್ಯಕ್ರಮಗಳು, ಸೌಕರ್ಯಗಳು ಅಥವಾ ಸಂಪರ್ಕ ಮಾಹಿತಿಯ ಬಗ್ಗೆ ಕೇಳಿ.",
            
            'ml': f"എനിക്ക് '{query}' സംബന്ധിച്ച് നിർദ്ദിഷ്ട വിവരങ്ങൾ ഇല്ല. ദയവായി പ്രവേശനം, ഫീസ്, ഹോസ്റ്റൽ, പ്ലെയ്സ്മെന്റുകൾ, പ്രോഗ്രാമുകൾ, സൗകര്യങ്ങൾ അല്ലെങ്കിൽ ബന്ധപ്പെടൽ വിവരങ്ങളെക്കുറിച്ച് ചോദിക്കുക.",
        }
        
        return responses.get(language, responses['en'])
    
    def _get_acknowledgment(self, language: str) -> str:
        """Get acknowledgment message"""
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
        """Get closing message"""
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
        """Get information about RAG system"""
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
