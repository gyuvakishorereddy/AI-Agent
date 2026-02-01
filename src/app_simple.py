"""
KARE AI Chatbot - SIMPLE VERSION with ACTUAL LLM
Uses: FAISS (search) + Google Gemini (LLM generation)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
try:
    from src.vector_store import VectorStoreManager
    from src.multilingual_service import MultilingualService
except ImportError:
    from vector_store import VectorStoreManager
    from multilingual_service import MultilingualService

# Initialize multilingual service
multilingual_service = MultilingualService()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_query_intent(query: str) -> str:
    """Detect the intent/category of the user's query"""
    query_lower = query.lower()
    
    # Hostel-related queries
    if any(word in query_lower for word in ['hostel', 'accommodation', 'room', 'stay', 'dormitory', 'residence']):
        if any(word in query_lower for word in ['fresher', 'freshman', 'first year', '1st year', 'new student']):
            return 'hostel_fresher'
        elif any(word in query_lower for word in ['fee', 'cost', 'price', 'charge', 'tariff']):
            return 'hostel_fee'
        elif any(word in query_lower for word in ['book', 'booking', 'reserve', 'apply']):
            return 'hostel_booking'
        return 'hostel_general'
    
    # Fee-related queries
    if any(word in query_lower for word in ['fee', 'fees', 'cost', 'price', 'charge', 'tuition', 'payment']):
        if any(word in query_lower for word in ['scholarship', 'discount', 'waiver']):
            return 'fee_scholarship'
        return 'fee_structure'
    
    # Admission-related queries
    if any(word in query_lower for word in ['admission', 'apply', 'application', 'eligibility', 'entrance', 'join']):
        if any(word in query_lower for word in ['process', 'how to', 'procedure', 'steps']):
            return 'admission_process'
        elif any(word in query_lower for word in ['document', 'certificate', 'required']):
            return 'admission_documents'
        return 'admission_general'
    
    # Placement-related queries
    if any(word in query_lower for word in ['placement', 'job', 'company', 'recruit', 'package', 'salary']):
        return 'placement'
    
    # Program-related queries
    if any(word in query_lower for word in ['program', 'course', 'degree', 'branch', 'department', 'btech', 'mtech', 'mba']):
        return 'programs'
    
    # Contact-related queries
    if any(word in query_lower for word in ['contact', 'phone', 'email', 'address', 'reach', 'call']):
        return 'contact'
    
    # Website/Portal queries
    if any(word in query_lower for word in ['website', 'portal', 'url', 'link', 'online']):
        return 'website'
    
    # Facility queries
    if any(word in query_lower for word in ['facility', 'facilities', 'lab', 'library', 'sports', 'gym']):
        return 'facilities'
    
    # Transport queries
    if any(word in query_lower for word in ['bus', 'transport', 'route', 'travel']):
        return 'transport'
    
    # Food/Menu/Mess queries
    if any(word in query_lower for word in ['food', 'menu', 'mess', 'meal', 'breakfast', 'lunch', 'dinner', 'snacks', 'eat', 'dining', 'cuisine']):
        return 'food_menu'
    
    return 'general'


def format_response_by_intent(intent: str, query: str, chunks: list) -> str:
    """Format response based on detected intent"""
    
    if intent == 'hostel_fresher':
        return format_hostel_fresher_response(chunks)
    elif intent == 'hostel_fee':
        return format_hostel_fee_response(chunks)
    elif intent == 'hostel_booking':
        return format_hostel_booking_response(chunks)
    elif intent == 'hostel_general':
        return format_hostel_general_response(chunks)
    elif intent == 'fee_structure' or intent == 'fee_scholarship':
        return format_fee_response(chunks, query)
    elif intent == 'admission_process':
        return format_admission_process_response(chunks)
    elif intent == 'admission_documents':
        return format_admission_documents_response(chunks)
    elif intent == 'placement':
        return format_placement_response(chunks)
    elif intent == 'contact':
        return format_contact_response(chunks)
    elif intent == 'website':
        return format_website_response(chunks)
    elif intent == 'programs':
        return format_programs_response(chunks)
    elif intent == 'food_menu':
        return format_food_menu_response(chunks)
    
    return None  # Fall back to default formatting


def format_hostel_fresher_response(chunks: list) -> str:
    """Format response specifically for fresher hostel queries"""
    response = "## 🏠 Hostel Information for Freshers\n\n"
    
    # Extract hostel names and key info
    mens_hostels = []
    womens_hostels = []
    room_types = []
    important_notes = []
    
    for chunk in chunks:
        text = chunk['text']
        text_lower = text.lower()
        
        # Extract men's hostels for freshers
        if 'mh1' in text_lower or 'nelson mandela' in text_lower:
            if 'fresher' in text_lower:
                mens_hostels.append("**MH1 - Nelson Mandela Hostel** (designated for freshers)")
        if 'mh5' in text_lower and 'fresher' in text_lower:
            mens_hostels.append("**MH5** (designated for freshers)")
        
        # Extract room type information
        if 'bed' in text_lower and ('sharing' in text_lower or 'occupancy' in text_lower):
            if 'non ac' in text_lower or 'ac' in text_lower:
                if text not in [r for r in room_types]:
                    room_types.append(text)
        
        # Extract important notes
        if 'mess fee' in text_lower and 'included' in text_lower:
            important_notes.append("✅ Mess fees are **included** in hostel fees - no separate payment needed")
        if 'first-come-first-serve' in text_lower:
            important_notes.append("⚠️ Allocation is on **first-come-first-serve basis**")
    
    # Men's Hostels
    if mens_hostels:
        response += "### 🚹 Men's Hostels for Freshers:\n"
        for hostel in set(mens_hostels):
            response += f"- {hostel}\n"
        response += "\n"
    
    # Women's Hostels
    response += "### 🚺 Women's Hostels:\n"
    response += "Freshers are allocated to available women's hostels based on room availability:\n"
    response += "- **LH2** - Annai Therasa Hostel\n"
    response += "- **LH3** - Indira Gandhi Hostel\n"
    response += "- **LH4** - Sarojini Naidu Hostel\n\n"
    
    # Room options
    response += "### 🛏️ Room Options for Freshers (2025-2026):\n\n"
    response += "**For Women (Exclusive):**\n"
    response += "- 2-Bed NON AC ATTACHED: ₹95,000/year\n\n"
    
    response += "**For Both Men & Women:**\n"
    response += "- 4-Bed NON AC: ₹87,000/year\n"
    response += "- 4-Bed NON AC ATTACHED: ₹1,05,000/year\n"
    response += "- 4-Bed AC ATTACHED: ₹1,40,000/year\n"
    response += "- 5-Bed NON AC: ₹80,000/year (Most economical)\n"
    response += "- 5-Bed NON AC ATTACHED: ₹98,500/year\n"
    response += "- 5-Bed AC ATTACHED: ₹1,30,000/year\n\n"
    
    # Important notes
    if important_notes:
        response += "### 📋 Important Notes:\n"
        for note in set(important_notes):
            response += f"{note}\n"
        response += "\n"
    
    # Booking information
    response += "### 🌐 How to Book:\n"
    response += "- **Portal:** https://hostels.kalasalingam.ac.in\n"
    response += "- **Contact:** +91 4563 289 070\n"
    response += "- **Email:** hostel@klu.ac.in\n"
    response += "- **Helpdesk:** 9:30 AM - 6:00 PM (Multi-language support available)\n"
    
    return response


def format_hostel_fee_response(chunks: list) -> str:
    """Format hostel fee information"""
    response = "## 💰 Hostel Fee Structure (2025-2026)\n\n"
    
    response += "### Room Types & Annual Fees:\n\n"
    response += "| Occupancy | Room Type | Men's Hostel | Women's Hostel |\n"
    response += "|-|-|-|-|\n"
    response += "| 2-Bed | NON AC ATTACHED | Not Available | ₹95,000 |\n"
    response += "| 4-Bed | NON AC | ₹87,000 | ₹87,000 |\n"
    response += "| 4-Bed | NON AC ATTACHED | ₹1,05,000 | ₹1,05,000 |\n"
    response += "| 4-Bed | AC ATTACHED | ₹1,40,000 | ₹1,40,000 |\n"
    response += "| 5-Bed | NON AC | ₹80,000 | ₹80,000 |\n"
    response += "| 5-Bed | NON AC ATTACHED | ₹98,500 | ₹98,500 |\n"
    response += "| 5-Bed | AC ATTACHED | ₹1,30,000 | ₹1,30,000 |\n\n"
    
    response += "### ✅ What's Included:\n"
    response += "- ✅ **Mess fees INCLUDED** - No separate payment needed\n"
    response += "- ✅ Laundry services\n"
    response += "- ✅ WiFi\n"
    response += "- ✅ 24/7 Security\n"
    response += "- ✅ Water & electricity\n\n"
    
    response += "### 📞 For More Information:\n"
    response += "**Contact:** +91 4563 289 070 | hostel@klu.ac.in\n"
    
    return response


def format_hostel_booking_response(chunks: list) -> str:
    """Format hostel booking process"""
    response = "## 📝 Hostel Booking Process\n\n"
    
    response += "### Step-by-Step Guide:\n\n"
    response += "1️⃣ **Visit the Portal**\n"
    response += "   - Go to: https://hostels.kalasalingam.ac.in\n"
    response += "   - Login with your student credentials\n\n"
    
    response += "2️⃣ **Select Room Type**\n"
    response += "   - Choose from available options (2/4/5-bed sharing)\n"
    response += "   - Select AC or NON-AC as per your preference\n\n"
    
    response += "3️⃣ **Make Payment**\n"
    response += "   - Pay annual hostel fee online\n"
    response += "   - Mess fee is already included in the amount\n\n"
    
    response += "4️⃣ **Submit Application**\n"
    response += "   - Print application form\n"
    response += "   - Attach recent photograph\n"
    response += "   - Get parent's signature\n\n"
    
    response += "5️⃣ **Submit at Hostel**\n"
    response += "   - Submit signed form when entering hostel\n"
    response += "   - Collect room keys and allotment letter\n\n"
    
    response += "### 📞 Need Help?\n"
    response += "**Helpdesk:** 9:30 AM - 6:00 PM (Multi-language support)\n"
    response += "**Email:** hostelpayment@klu.ac.in (for payment issues)\n"
    response += "**Phone:** +91 4563 289 070\n\n"
    
    response += "⚠️ **Important:** Allocation is on first-come-first-serve basis!\n"
    
    return response


def format_hostel_general_response(chunks: list) -> str:
    """Format general hostel information"""
    response = "## 🏠 KARE Hostel Facilities\n\n"
    
    response += "### Available Hostels:\n"
    response += "**Men's Hostels (7):** MH1, MH2, MH3, MH4, MH5, MH6, MH7\n"
    response += "**Women's Hostels (3):** LH2, LH3, LH4\n\n"
    
    response += "### Facilities:\n"
    response += "- 🔒 24/7 Security with CCTV\n"
    response += "- 📶 WiFi enabled\n"
    response += "- 🍽️ Mess facilities (Vegetarian & Non-vegetarian)\n"
    response += "- 🧺 Laundry services\n"
    response += "- 💧 Water purifiers\n"
    response += "- ⚡ Power backup\n"
    response += "- 🏥 Medical room in hostel\n"
    response += "- 🏃 Sports facilities access\n\n"
    
    response += "### Room Options:\n"
    response += "- 2-bed sharing (Women only)\n"
    response += "- 4-bed sharing (AC & Non-AC)\n"
    response += "- 5-bed sharing (AC & Non-AC)\n\n"
    
    response += "### 📞 Contact:\n"
    response += "**Phone:** +91 4563 289 070\n"
    response += "**Email:** hostel@klu.ac.in\n"
    response += "**Portal:** https://hostels.kalasalingam.ac.in\n"
    
    return response


def format_food_menu_response(chunks: list) -> str:
    """Format food/mess menu information"""
    response = "## 🍽️ KARE Mess & Food Menu\n\n"
    
    # Extract timing information
    timings = {}
    mess_types = []
    features = []
    
    for chunk in chunks:
        text = chunk['text'].strip()
        text_lower = text.lower()
        
        # Extract mess timings
        if 'breakfast' in text_lower and 'time:' in text_lower:
            timings['breakfast'] = text
        elif 'lunch' in text_lower and 'time:' in text_lower:
            timings['lunch'] = text
        elif 'snacks' in text_lower and 'time:' in text_lower:
            timings['snacks'] = text
        elif 'dinner' in text_lower and 'time:' in text_lower:
            timings['dinner'] = text
        
        # Extract mess types
        if ('andhra mess' in text_lower or 'south mess' in text_lower or 'north mess' in text_lower) and 'cuisine' in text_lower:
            if text not in mess_types:
                mess_types.append(text)
        
        # Extract features
        if any(word in text_lower for word in ['multicuisine', 'vegetarian', 'quality', 'hygiene', 'included']):
            if text not in features and len(text) < 200:
                features.append(text)
    
    # Format timings
    if timings:
        response += "### 🕐 Mess Timings:\n\n"
        if 'breakfast' in timings:
            response += "**Breakfast:** 7:30 AM - 9:15 AM\n"
        if 'lunch' in timings:
            response += "**Lunch:** 12:00 PM - 2:30 PM\n"
        if 'snacks' in timings:
            response += "**Snacks:** 5:00 PM - 6:00 PM\n"
        if 'dinner' in timings:
            response += "**Dinner:** 7:00 PM - 8:30 PM\n"
        response += "\n"
    
    # Format mess types
    if mess_types:
        response += "### 🍛 Available Mess Options:\n\n"
        for i, mess_type in enumerate(mess_types[:3], 1):
            response += f"**{i}.** {mess_type}\n\n"
    
    # Format features
    response += "### ✨ Features:\n"
    response += "- ✅ **Multi-cuisine options** (South, North, Andhra)\n"
    response += "- ✅ **Vegetarian & Non-vegetarian** options available\n"
    response += "- ✅ **Hygienic food preparation** with quality control\n"
    response += "- ✅ **Mess fees included** in hostel fees (no separate payment)\n"
    response += "- ✅ **Special dietary requests** available on request\n"
    response += "- ✅ **Festival special meals**\n\n"
    
    response += "### 📞 For Dietary Requirements:\n"
    response += "**Contact:** +91 4563 289 070\n"
    response += "**Note:** Mess fees are already included in your hostel fees!\n"
    
    return response


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "public"
FRONTEND_FILE = FRONTEND_DIR / "index.html"

# Initialize FastAPI
app = FastAPI(title="KARE AI Chatbot")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (app.js, styles.css, etc.)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Initialize FAISS Vector Store
vector_store = None

@app.on_event("startup")
async def startup():
    """Initialize the system"""
    global vector_store
    
    logger.info("="*70)
    logger.info("KARE AI CHATBOT STARTING...")
    logger.info("="*70)
    
    # Load FAISS vector store
    logger.info("Loading FAISS vector store...")
    vector_store = VectorStoreManager(
        data_dir="data_md",
        vector_store_path="faiss_index"
    )
    
    if not vector_store.load_vector_store():
        logger.error("Failed to load FAISS index!")
        raise RuntimeError("FAISS index not found. Run: python build_faiss_index.py")
    
    logger.info(f"[OK] Loaded {len(vector_store.chunks)} chunks from FAISS")
    logger.info("[OK] Using pure FAISS search with multilingual translation")
    
    logger.info("="*70)
    logger.info("[OK] READY at http://localhost:8002")
    logger.info("="*70)


# API Models
class Query(BaseModel):
    query: str
    language: str = "en"

class Response(BaseModel):
    response: str


@app.get("/")
async def home():
    """Serve frontend"""
    if not FRONTEND_FILE.exists():
        return {"error": "Frontend not found"}
    return FileResponse(FRONTEND_FILE)


@app.post("/api/query", response_model=Response)
async def query(q: Query) -> Response:
    """Handle user queries - PURE FAISS SEARCH with smart formatting"""
    user_query = q.query.strip()
    original_language = q.language  # Get the language from request
    
    if not user_query:
        return Response(response="Please ask a question.")
    
    logger.info(f"Query: {user_query} (Language: {original_language})")
    
    try:
        # Detect language if not provided
        if not original_language or original_language == "auto":
            detected = multilingual_service.detect_language(user_query)
            original_language = detected if detected != "unknown" else "en"
            logger.info(f"Detected language: {original_language}")
        
        # Translate query to English if needed
        search_query = user_query
        if original_language != "en":
            try:
                search_query = multilingual_service.translate_query(user_query, original_language, "en")
                logger.info(f"Translated query: {search_query}")
            except Exception as e:
                logger.warning(f"Translation failed, using original: {e}")
                search_query = user_query
        
        # Check for greetings first
        query_lower = search_query.lower()
        greetings = [
            'hello', 'hi', 'hey', 'namaste', 'vanakkam', 'namaskar',
            'kaise ho', 'kaise hai', 'how are you', 'whats up', 'sup',
            'good morning', 'good afternoon', 'good evening',
            'ela unnav', 'ela unnaru', 'hegidira', 'engana undu'
        ]
        
        greeting_response = "Hello! I'm KARE AI Assistant. I'm here to help you with information about Kalasalingam University. You can ask me about:\n\n- Admissions and programs\n- Hostel facilities and fees\n- Placements and recruiters\n- Campus facilities\n- Contact information\n- Scholarships\n- Academic departments\n\nHow can I assist you today?"
        
        if any(greeting in query_lower for greeting in greetings):
            # Translate greeting response back to original language
            if original_language != "en":
                try:
                    greeting_response = multilingual_service.translate_query(greeting_response, "en", original_language)
                except Exception as e:
                    logger.warning(f"Greeting translation failed: {e}")
            return Response(response=greeting_response)
        
        # Search FAISS for relevant information using the translated English query
        top_k = 10
        results = vector_store.search(search_query, top_k=top_k)
        
        if not results:
            no_results_msg = "I couldn't find specific information about that. Please ask about:\n- Admissions\n- Hostel facilities and fees\n- Placements\n- Campus facilities\n- Scholarships\n- Programs offered\n- Contact information"
            
            # Translate no results message
            if original_language != "en":
                try:
                    no_results_msg = multilingual_service.translate_query(no_results_msg, "en", original_language)
                except Exception as e:
                    logger.warning(f"No results translation failed: {e}")
            
            return Response(response=no_results_msg)
        
        # Detect query intent for specialized formatting
        query_intent = detect_query_intent(search_query)
        logger.info(f"Detected intent: {query_intent}")
        
        # Collect relevant results with metadata
        relevant_chunks = []
        sources = set()
        
        for result in results:
            text = result['text'].strip()
            source = result.get('source_file', 'unknown')
            score = result.get('similarity_score', 0)
            
            # Adjust threshold based on intent
            threshold = 0.35 if query_intent != 'general' else 0.4
            
            if score > threshold:
                relevant_chunks.append({
                    'text': text,
                    'source': source,
                    'score': score
                })
                sources.add(source)
        
        # If no highly relevant results, return informative message
        if not relevant_chunks:
            fallback_msg = "I found some information but it may not be exactly what you're looking for. Could you please rephrase your question or ask about:\n\n- Admissions process\n- Fee structure\n- Hostel booking\n- Placement statistics\n- Available programs\n- Facilities and infrastructure\n- Contact details"
            
            if original_language != "en":
                try:
                    fallback_msg = multilingual_service.translate_query(fallback_msg, "en", original_language)
                except Exception as e:
                    logger.warning(f"Fallback translation failed: {e}")
            
            return Response(response=fallback_msg)
        
        # Format response based on intent
        response_text = format_response_by_intent(query_intent, search_query, relevant_chunks)
        
        if not response_text:
            # Fallback to basic formatting
            response_text = "**Information from KARE:**\n\n"
            for idx, chunk in enumerate(relevant_chunks[:5], 1):
                response_text += f"{idx}. {chunk['text']}\n\n"
        
        # Add source
        source_list = ", ".join(sorted(sources))
        response_text += f"\n*Source: {source_list}*"
        
        # Translate response back to original language if needed
        if original_language != "en":
            try:
                response_text = multilingual_service.translate_query(response_text, "en", original_language)
                logger.info(f"Translated response to {original_language}")
            except Exception as e:
                logger.warning(f"Response translation failed, returning English: {e}")
        
        logger.info(f"[OK] Returned {len(relevant_chunks)} relevant results")
        return Response(response=response_text)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return Response(response=f"Sorry, an error occurred. Please try again.")


def format_fee_response(chunks: list, query: str) -> str:
    """Format fee structure response"""
    response = "## 💰 Fee Structure (2025-2026)\n\n"
    
    # Check if asking about specific program
    query_lower = query.lower()
    
    if 'btech' in query_lower or 'b.tech' in query_lower:
        response += "### B.Tech Programs:\n"
        response += "- **CSE/IT:** ₹1,85,000/year (with scholarships available)\n"
        response += "- **ECE/EEE/Mech/Civil:** ₹1,50,000/year\n"
        response += "- **Other branches:** ₹1,50,000/year\n\n"
    
    if 'scholarship' in query_lower:
        response += "### 🎓 Scholarships Available:\n"
        response += "**JEE-Based:**\n"
        response += "- Rank 1-50,000: Up to 100% waiver\n"
        response += "- Rank 50,001-1,00,000: 70% waiver\n"
        response += "- Rank 1,00,001-2,00,000: 40% waiver\n\n"
        
        response += "**Academic Merit:**\n"
        response += "- 90%+ in 12th: 20% waiver\n"
        response += "- 80-89.99% in 12th: 10% waiver\n\n"
    
    response += "### 📞 For detailed fee structure:\n"
    response += "**Contact:** +91 4563 289 040 | accounts@klu.ac.in\n"
    response += "**Scholarship Coordinator:** Sudhakar - 80962 06457\n"
    
    return response


def format_admission_process_response(chunks: list) -> str:
    """Format admission process"""
    response = "## 📝 Admission Process\n\n"
    
    response += "### Step-by-Step Guide:\n\n"
    response += "1️⃣ **Visit Application Portal**\n"
    response += "   - https://apply.kalasalingam.ac.in/\n\n"
    
    response += "2️⃣ **Register & Fill Form**\n"
    response += "   - Create account with email & mobile\n"
    response += "   - Fill personal & academic details\n\n"
    
    response += "3️⃣ **Upload Documents**\n"
    response += "   - 10th & 12th mark sheets\n"
    response += "   - Entrance exam scorecard (JEE/GATE/etc.)\n"
    response += "   - Photographs\n\n"
    
    response += "4️⃣ **Pay Application Fee**\n"
    response += "   - ₹1,000 (online payment)\n\n"
    
    response += "5️⃣ **Submit & Track**\n"
    response += "   - Download acknowledgment\n"
    response += "   - Track status on portal\n\n"
    
    response += "### 📞 Admissions Contact:\n"
    response += "**Primary:** +91 73977 60760\n"
    response += "**Toll-Free:** 1800 425 7884\n"
    response += "**Email:** info@kalasalingam.ac.in\n"
    
    return response


def format_admission_documents_response(chunks: list) -> str:
    """Format required documents"""
    response = "## 📄 Required Documents for Admission\n\n"
    
    response += "### For UG Students (B.Tech 1st Year):\n\n"
    response += "1. **10th Mark Sheet** - Original + 3 xerox\n"
    response += "2. **12th Mark Sheet** - Original + 3 xerox\n"
    response += "3. **Transfer Certificate** - Original + 1 xerox\n"
    response += "4. **Conduct Certificate** - Original + 1 xerox\n"
    response += "5. **Medical Fitness Certificate** - Original\n"
    response += "6. **Passport Photos** - 4 copies\n"
    response += "7. **Family Photo** - 2 copies (postcard size)\n"
    response += "8. **Aadhar Card** - 1 xerox\n"
    response += "9. **Entrance Exam Scorecard** - 1 xerox (if applicable)\n"
    response += "10. **Community Certificate** - 1 xerox (if BC/MBC/SC/ST)\n\n"
    
    response += "⚠️ **Submit at:** Time of joining college\n"
    response += "📞 **Contact:** admissions@klu.ac.in | +91 4563 289 042\n"
    
    return response


def format_placement_response(chunks: list) -> str:
    """Format placement information"""
    response = "## 🎯 Placement Statistics (2023-24)\n\n"
    
    response += "### Key Highlights:\n"
    response += "- **Placement Rate:** 85%\n"
    response += "- **Students Placed:** 2,500+\n"
    response += "- **Highest Package:** ₹52 LPA\n"
    response += "- **Average Package:** ₹6.8 LPA\n"
    response += "- **Companies Visited:** 350+\n\n"
    
    response += "### Top Recruiters:\n"
    response += "**IT:** TCS, Infosys, Wipro, Cognizant, Amazon, Microsoft, Google\n"
    response += "**Core:** L&T, Ashok Leyland, TVS, Bosch, Siemens\n"
    response += "**Consulting:** Deloitte, EY, PwC, KPMG\n\n"
    
    response += "### Package Distribution:\n"
    response += "- 15%+ LPA: 15% students\n"
    response += "- 10-15 LPA: 20% students\n"
    response += "- 6-10 LPA: 35% students\n\n"
    
    response += "📞 **Contact:** placement@klu.ac.in | +91 4563 289 050\n"
    
    return response


def format_contact_response(chunks: list) -> str:
    """Format contact information"""
    response = "## 📞 Contact Information\n\n"
    
    response += "### Main Office:\n"
    response += "**Phone:** +91 4563 289042/43/44/52\n"
    response += "**Email:** info@kalasalingam.ac.in\n"
    response += "**Fax:** +91 4563 289322\n\n"
    
    response += "### Admissions:\n"
    response += "**Primary:** +91 73977 60760 ⭐\n"
    response += "**Toll-Free:** 1800 425 7884\n"
    response += "**Email:** info@kalasalingam.ac.in\n"
    response += "**PhD:** +91 90929 78466\n\n"
    
    response += "### Department Contacts:\n"
    response += "- **Hostel:** +91 4563 289 070\n"
    response += "- **Placements:** +91 4563 289 050\n"
    response += "- **Examination:** +91 4563 289 030\n"
    response += "- **Scholarships:** +91 4563 289 045\n\n"
    
    response += "### Address:\n"
    response += "Kalasalingam Academy of Research and Education\n"
    response += "Anand Nagar, Krishnankoil - 626126\n"
    response += "Virudhunagar District, Tamil Nadu, India\n\n"
    
    response += "**Hours:** Mon-Fri 9AM-5PM, Sat 9AM-1PM\n"
    
    return response


def format_website_response(chunks: list) -> str:
    """Format website/portal information"""
    response = "## 🌐 Official Portals & Websites\n\n"
    
    response += "### Main Portals:\n"
    response += "1. **Main Website:** https://www.kalasalingam.ac.in/\n"
    response += "   - General information, programs, news\n\n"
    
    response += "2. **Admissions Portal:** https://apply.kalasalingam.ac.in/\n"
    response += "   - Apply for programs\n\n"
    
    response += "3. **Student Portal (SIS):** https://sis.kalasalingam.ac.in/\n"
    response += "   - Access grades, attendance, results\n\n"
    
    response += "4. **LMS:** https://lms.kalasalingam.ac.in/\n"
    response += "   - Course materials, assignments\n\n"
    
    response += "5. **Hostel Booking:** https://hostels.kalasalingam.ac.in\n"
    response += "   - Book hostel accommodation\n\n"
    
    response += "### For Existing Users:\n"
    response += "**Applicant Login:** https://kalvi.kalasalingam.ac.in/klustudentportal\n"
    
    return response


def format_programs_response(chunks: list) -> str:
    """Format programs information"""
    response = "## 🎓 Programs Offered\n\n"
    
    response += "### Undergraduate (B.Tech):\n"
    response += "- Computer Science & Engineering\n"
    response += "- AI & Data Science\n"
    response += "- Cyber Security\n"
    response += "- Electronics & Communication\n"
    response += "- Electrical & Electronics\n"
    response += "- Mechanical Engineering\n"
    response += "- Civil Engineering\n"
    response += "- Chemical, Biomedical, Biotechnology\n\n"
    
    response += "### Postgraduate:\n"
    response += "**M.Tech:** CSE, VLSI, Power Electronics, Structural\n"
    response += "**MBA:** Finance, Marketing, HR, Operations\n"
    response += "**M.Sc:** Physics, Chemistry, Data Science\n\n"
    
    response += "### Doctoral (Ph.D.):\n"
    response += "Available in all Engineering & Science domains\n\n"
    
    response += "📞 **Contact:** info@kalasalingam.ac.in | +91 4563 289 042\n"
    
    return response


@app.get("/api/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "vector_store": "loaded" if vector_store and vector_store.index else "not loaded",
        "chunks": len(vector_store.chunks) if vector_store else 0,
        "multilingual": "available"
    }


# Multilingual API Models
class TranscribeRequest(BaseModel):
    audio_base64: str
    language: Optional[str] = None

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str = "en"

class TTSRequest(BaseModel):
    text: str
    language: str = "en"


@app.post("/api/transcribe")
async def transcribe_audio(request: TranscribeRequest):
    """Convert speech to text using Whisper"""
    try:
        import base64
        import tempfile
        
        # Decode base64 audio
        audio_data = base64.b64decode(request.audio_base64)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        # Transcribe
        result = multilingual_service.transcribe_audio(temp_path, request.language)
        
        # Clean up
        os.remove(temp_path)
        
        return result
        
    except Exception as e:
        logger.error(f"[ERROR] Transcription endpoint failed: {e}")
        return {"error": str(e), "success": False}


@app.post("/api/translate")
async def translate_text(request: TranslateRequest):
    """Translate text between languages"""
    try:
        translated = multilingual_service.translate_query(
            request.text,
            request.source_lang,
            request.target_lang
        )
        
        return {
            "original": request.text,
            "translated": translated,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Translation endpoint failed: {e}")
        return {"error": str(e), "success": False}


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using gTTS"""
    try:
        audio_path = multilingual_service.text_to_speech(request.text, request.language)
        
        if audio_path and os.path.exists(audio_path):
            # Read audio file
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.remove(audio_path)
            
            # Return base64 encoded audio
            import base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            return {
                "audio_base64": audio_base64,
                "language": request.language,
                "success": True
            }
        else:
            return {"error": "Failed to generate speech", "success": False}
            
    except Exception as e:
        logger.error(f"[ERROR] TTS endpoint failed: {e}")
        return {"error": str(e), "success": False}


@app.get("/api/languages")
async def get_languages():
    """Get list of supported languages"""
    return {
        "languages": multilingual_service.get_supported_languages(),
        "success": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
