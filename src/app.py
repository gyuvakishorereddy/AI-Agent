"""
KARE AI Chatbot - FAISS RAG with Vector Store
Uses FAISS vector store with sentence-transformers embeddings
Routes all queries through RAG engine for intelligent response generation
Includes: MySQL-backed auth (signup/signin) and chat history storage
"""

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import io
import re
from pathlib import Path
import uvicorn
from typing import Optional

# gTTS for text-to-speech (free, no API key)
try:
    from gtts import gTTS
    _GTTS_AVAILABLE = True
except Exception:
    _GTTS_AVAILABLE = False

# Import RAG engine
try:
    from src.rag_engine import RAGResponseGenerator
except ImportError:
    from rag_engine import RAGResponseGenerator

# Import database module
try:
    from src.database import (
        init_database, signup_user, signin_user, validate_token, logout_user,
        get_or_create_session, save_message, get_user_chat_sessions,
        get_session_messages, delete_chat_session, create_chat_session,
    )
except ImportError:
    from database import (
        init_database, signup_user, signin_user, validate_token, logout_user,
        get_or_create_session, save_message, get_user_chat_sessions,
        get_session_messages, delete_chat_session, create_chat_session,
    )

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_FILE = BASE_DIR / "public" / "index.html"
LOGIN_FILE = BASE_DIR / "public" / "login.html"

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

# Mount static files
STATIC_DIR = BASE_DIR / "public"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global RAG instance
rag_engine = None

# --------------- Pydantic Models ---------------

class Query(BaseModel):
    query: str
    language: Optional[str] = "en"
    session_id: Optional[str] = None


class Response(BaseModel):
    response: str
    detected_language: Optional[str] = "en"


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class SigninRequest(BaseModel):
    email: str
    password: str


# --------------- Auth Helper ---------------

def _get_current_user(authorization: Optional[str] = None):
    """Extract and validate user from Authorization header."""
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    return validate_token(token)


def detect_language_backend(text: str, frontend_lang: str = "en") -> str:
    """
    Detect language on backend, handling Romanized Indian languages (Tinglish, Hinglish, etc.)
    that the frontend can't detect via Unicode ranges.
    """
    # If frontend already detected a non-English script, trust it
    if frontend_lang and frontend_lang != "en":
        return frontend_lang

    text_lower = text.lower().strip()

    # Hindi / Hinglish markers
    hindi_words = {
        'kya', 'hai', 'kaise', 'ho', 'hain', 'mein', 'mujhe', 'kahan', 'kitna', 'kitni',
        'aur', 'nahi', 'haan', 'accha', 'theek', 'batao', 'bataiye', 'chahiye',
        'karo', 'karna', 'kaisa', 'kaisi', 'yeh', 'woh', 'kab', 'kyun', 'kyu',
        'paisa', 'rupee', 'padhai', 'namaste', 'namaskar', 'dhanyavaad', 'shukriya',
        'bhai', 'yaar', 'ji', 'aap', 'tum', 'hum', 'unka', 'iska', 'ka', 'ki', 'ke',
        'se', 'ko', 'ne', 'par', 'pe', 'tak', 'abhi', 'bahut', 'zyada', 'kam',
        'milega', 'milta', 'dedo', 'dena', 'lena', 'jana', 'aana', 'jaana',
    }

    # Telugu / Tinglish markers
    telugu_words = {
        'ela', 'unnav', 'unnaru', 'undi', 'ento', 'em', 'emi', 'cheppandi', 'cheppu',
        'gurinchi', 'kosam', 'ante', 'aythe', 'kadu', 'avunu', 'ledu', 'undhi',
        'meeru', 'nenu', 'manamu', 'vaadu', 'aameku', 'ekkada', 'entha', 'enduku',
        'ela', 'epudu', 'evaru', 'chala', 'bagundi', 'baaga', 'manchidi',
        'namaskaram', 'namasthe', 'dhanyavaadalu', 'andi', 'garu',
        'padandi', 'ivvandi', 'cheyandi', 'randi', 'vellandi',
        'hostel', 'fees', 'admission', 'college', 'bus',
        'kavali', 'kaavali', 'dorukutunda', 'cheppu', 'teliyali',
    }

    # Tamil / Tanglish markers
    tamil_words = {
        'eppadi', 'irukka', 'irukkinga', 'irukku', 'enna', 'ennaku', 'enaku',
        'sollunga', 'solluga', 'theriyum', 'theriyala', 'theriyadhu',
        'vanakkam', 'nandri', 'aamaa', 'illa', 'irukku', 'illai',
        'yenna', 'yaar', 'enga', 'evvalavu', 'yeppo', 'yen', 'yeppadi',
        'nalla', 'romba', 'konjam', 'podu', 'vaa', 'ponga', 'vaanga',
        'padipu', 'kattalai', 'viduthi', 'kattanam',
        'naan', 'nee', 'avar', 'aval', 'ungal', 'engal', 'idhu',
        'venum', 'vendaam', 'mudiyum', 'mudiyala',
    }

    words = set(text_lower.replace('?', ' ').replace('!', ' ').replace(',', ' ').split())

    hindi_count = len(words & hindi_words)
    telugu_count = len(words & telugu_words)
    tamil_count = len(words & tamil_words)

    # Need at least 1 strong match for short queries, 2 for longer
    threshold = 1 if len(words) <= 4 else 2

    max_count = max(hindi_count, telugu_count, tamil_count)
    if max_count >= threshold:
        if hindi_count == max_count:
            return "hi"
        elif telugu_count == max_count:
            return "te"
        elif tamil_count == max_count:
            return "ta"

    return frontend_lang or "en"


@app.on_event("startup")
async def startup():
    global rag_engine

    logger.info("=" * 70)
    logger.info("KARE AI CHATBOT STARTING...")
    logger.info("=" * 70)

    # Initialize MySQL database
    logger.info("Initializing MySQL database...")
    if init_database():
        logger.info("✅ MySQL database ready")
    else:
        logger.error("❌ MySQL database initialization failed!")

    try:
        logger.info("Initializing FAISS RAG Engine...")
        rag_engine = RAGResponseGenerator(
            vector_store_path=str(BASE_DIR / "faiss_index"),
            data_dir=str(BASE_DIR / "data_md"),
            use_llm=False,
        )

        if rag_engine.vector_store and rag_engine.vector_store.index:
            logger.info(f"Loaded {len(rag_engine.vector_store.chunks)} chunks from FAISS")
            logger.info("Ready at http://localhost:8000")
        else:
            logger.error("FAISS index not loaded!")

    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        import traceback
        traceback.print_exc()

    logger.info("=" * 70)


# =============== Page Routes ===============

@app.get("/")
async def root():
    """Serve login page as default landing page."""
    return FileResponse(LOGIN_FILE)


@app.get("/login")
async def login_page():
    return FileResponse(LOGIN_FILE)


@app.get("/chat")
async def chat_page():
    return FileResponse(FRONTEND_FILE)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_initialized": rag_engine is not None,
        "faiss_loaded": rag_engine and rag_engine.vector_store and rag_engine.vector_store.index is not None,
        "chunks": len(rag_engine.vector_store.chunks) if rag_engine and rag_engine.vector_store else 0,
    }


# =============== Auth Endpoints ===============

@app.post("/api/signup")
async def api_signup(req: SignupRequest):
    result = signup_user(req.name, req.email, req.password)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/signin")
async def api_signin(req: SigninRequest):
    result = signin_user(req.email, req.password)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["message"])
    return result


@app.post("/api/logout")
async def api_logout(request: Request):
    auth = request.headers.get("Authorization", "")
    token = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else auth
    logout_user(token)
    return {"success": True, "message": "Logged out"}


@app.get("/api/me")
async def api_me(request: Request):
    auth = request.headers.get("Authorization", "")
    user = _get_current_user(auth)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"success": True, "user": user}


# =============== Chat Session Endpoints ===============

@app.get("/api/chat/sessions")
async def api_get_sessions(request: Request):
    auth = request.headers.get("Authorization", "")
    user = _get_current_user(auth)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    sessions = get_user_chat_sessions(user["id"])
    return {"success": True, "sessions": sessions}


@app.get("/api/chat/sessions/{session_id}/messages")
async def api_get_messages(session_id: int, request: Request):
    auth = request.headers.get("Authorization", "")
    user = _get_current_user(auth)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    messages = get_session_messages(session_id, user["id"])
    return {"success": True, "messages": messages}


@app.delete("/api/chat/sessions/{session_id}")
async def api_delete_session(session_id: int, request: Request):
    auth = request.headers.get("Authorization", "")
    user = _get_current_user(auth)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    deleted = delete_chat_session(session_id, user["id"])
    return {"success": deleted}


# =============== Query Endpoint (with chat storage) ===============


@app.post("/api/query")
async def query(q: Query, request: Request) -> Response:
    """Route all queries through RAG engine and store chat in MySQL."""
    user_query = q.query.strip()
    frontend_lang = q.language or "en"
    session_id = q.session_id or "default"

    logger.info(f"Query: {user_query} | Frontend lang: {frontend_lang}")

    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    if not rag_engine.vector_store or not rag_engine.vector_store.index:
        raise HTTPException(status_code=503, detail="FAISS index not loaded")

    # Get current user (if authenticated)
    auth = request.headers.get("Authorization", "")
    user = _get_current_user(auth)

    try:
        # Detect language (handles Tinglish, Hinglish, Tanglish etc.)
        detected_lang = detect_language_backend(user_query, frontend_lang)
        logger.info(f"Detected language: {detected_lang}")

        # Route through RAG engine — handles greetings, relevance filtering, formatting
        response_text = rag_engine.generate_response(
            query=user_query,
            language=detected_lang,
            top_k=5,
        )

        # Store chat in MySQL if user is authenticated
        if user:
            try:
                db_session_id = get_or_create_session(user["id"], session_id)
                if db_session_id:
                    save_message(user["id"], db_session_id, "user", user_query, detected_lang)
                    save_message(user["id"], db_session_id, "bot", response_text, detected_lang)
            except Exception as db_err:
                logger.error(f"DB save error (non-fatal): {db_err}")

        logger.info(f"Response generated ({len(response_text)} chars) for: '{user_query}'")
        return Response(response=response_text, detected_language=detected_lang)

    except Exception as e:
        logger.error(f"Query error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============== Text-to-Speech (gTTS — free, no API key) ===============

class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "en"


_GTTS_LANG_MAP = {
    "en": "en", "hi": "hi", "te": "te",
    "ta": "ta", "kn": "kn", "ml": "ml",
}


@app.post("/api/tts")
async def api_tts(req: TTSRequest):
    """Convert text to speech using gTTS and return MP3."""
    if not _GTTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="gTTS not installed")

    lang_code = _GTTS_LANG_MAP.get(req.language or "en", "en")

    # Strip markdown symbols that TTS would read literally
    clean_text = re.sub(r'[*_`#]', '', req.text)
    clean_text = clean_text.replace('•', '').replace('\\n', ' ').strip()

    try:
        buf = io.BytesIO()
        tts = gTTS(text=clean_text[:3000], lang=lang_code, slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=tts.mp3"},
        )
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
