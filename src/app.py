"""
KARE AI Chatbot - FAISS RAG with Vector Store
Uses FAISS vector store with sentence-transformers embeddings
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging
from pathlib import Path
import uvicorn
from typing import Optional

# Import RAG engine
try:
    from src.rag_engine import RAGResponseGenerator
except ImportError:
    from rag_engine import RAGResponseGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_FILE = BASE_DIR / "public" / "index.html"

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

# Global RAG instance
rag_engine = None

# Models
class Query(BaseModel):
    query: str
    session_id: Optional[str] = None


class Response(BaseModel):
    response: str


@app.on_event("startup")
async def startup():
    global rag_engine
    
    logger.info("="*70)
    logger.info("KARE AI CHATBOT STARTING...")
    logger.info("="*70)
    
    try:
        # Initialize RAG engine
        logger.info("Initializing FAISS RAG Engine...")
        rag_engine = RAGResponseGenerator(
            vector_store_path=str(BASE_DIR / "faiss_index"),
            data_dir=str(BASE_DIR / "data_md"),
            use_llm=False  # Don't use LLM for faster responses
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
    
    logger.info("="*70)


# Routes
@app.get("/")
async def root():
    return FileResponse(FRONTEND_FILE)


@app.get("/styles.css")
async def styles():
    return FileResponse(BASE_DIR / "public" / "styles.css")


@app.get("/app.js")
async def app_js():
    return FileResponse(BASE_DIR / "public" / "app.js")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_initialized": rag_engine is not None,
        "faiss_loaded": rag_engine and rag_engine.vector_store and rag_engine.vector_store.index is not None,
        "chunks": len(rag_engine.vector_store.chunks) if rag_engine and rag_engine.vector_store else 0
    }


@app.post("/api/query")
async def query(q: Query) -> Response:
    """Query using FAISS RAG engine"""
    user_query = q.query
    
    logger.info(f"Query: {user_query}")
    
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    if not rag_engine.vector_store or not rag_engine.vector_store.index:
        raise HTTPException(status_code=503, detail="FAISS index not loaded")
    
    try:
        query_lower = user_query.lower()
        
        # Intelligent query type detection
        query_keywords = {
            'fee': ['fee', 'fees', 'cost', 'price', 'how much', 'amount', 'charge'],
            'website': ['website', 'link', 'url', 'portal', 'online', 'web'],
            'booking': ['book', 'booking', 'reserve', 'reservation', 'apply', 'application'],
            'admission': ['admission', 'admit', 'entrance', 'eligibility', 'join'],
            'program': ['program', 'course', 'degree', 'b.tech', 'm.tech', 'mba', 'offered'],
            'placement': ['placement', 'job', 'company', 'recruit', 'salary', 'package'],
            'hostel': ['hostel', 'accommodation', 'room', 'stay'],
            'contact': ['contact', 'phone', 'email', 'call', 'reach'],
            'how': ['how', 'process', 'procedure', 'steps', 'way']
        }
        
        # Detect query intent
        detected_intents = []
        for intent, keywords in query_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_intents.append(intent)
        
        # Adjust search parameters based on intent
        if 'fee' in detected_intents:
            top_k = 15
        elif 'booking' in detected_intents or 'how' in detected_intents:
            top_k = 10
        else:
            top_k = 8
        
        # Search FAISS vector store
        results = rag_engine.vector_store.search(user_query, top_k=top_k)
        
        if results:
            response_text = ""
            
            # Smart formatting based on detected intent
            if 'booking' in detected_intents and 'hostel' in detected_intents:
                # Hostel booking process
                response_text = "**Hostel Booking Process:**\n\n"
                
                # Extract relevant information
                process_steps = []
                portal_info = []
                requirements = []
                contact_info = []
                
                for result in results:
                    text = result['text'].strip()
                    text_lower = text.lower()
                    
                    if 'portal' in text_lower or 'website' in text_lower or 'http' in text_lower:
                        if text not in portal_info and len(portal_info) < 2:
                            portal_info.append(text)
                    elif 'requirement' in text_lower or 'document' in text_lower or 'form' in text_lower:
                        if text not in requirements and len(requirements) < 3:
                            requirements.append(text)
                    elif 'phone' in text_lower or 'email' in text_lower:
                        if text not in contact_info and len(contact_info) < 1:
                            contact_info.append(text)
                    elif any(word in text_lower for word in ['step', 'process', 'apply', 'submit']):
                        if text not in process_steps and len(process_steps) < 5:
                            process_steps.append(text)
                
                # Format response
                if portal_info:
                    response_text += "**🌐 Booking Portal:**\n"
                    for info in portal_info:
                        response_text += f"{info}\n\n"
                
                if process_steps:
                    response_text += "**📋 Process:**\n"
                    for i, step in enumerate(process_steps, 1):
                        response_text += f"{i}. {step}\n\n"
                
                if requirements:
                    response_text += "**📄 Requirements:**\n"
                    for req in requirements:
                        response_text += f"• {req}\n\n"
                
                if contact_info:
                    response_text += "**📞 Contact:**\n"
                    response_text += f"{contact_info[0]}\n\n"
            
            elif 'fee' in detected_intents and 'hostel' in detected_intents:
                # Hostel fees
                response_text = "**Hostel Fee Structure (2025-2026):**\n\n"
                
                fee_items = []
                notes = []
                contact = []
                
                for result in results:
                    text = result['text'].strip()
                    text_lower = text.lower()
                    
                    if 'phone' in text_lower and 'email' in text_lower:
                        if text not in contact:
                            contact.append(text)
                    elif any(word in text_lower for word in ['note:', 'important', 'included', 'mess fees']):
                        if text not in notes and len(notes) < 3:
                            notes.append(text)
                    elif any(char.isdigit() for char in text) and len([c for c in text if c.isdigit()]) >= 4:
                        if text not in fee_items:
                            fee_items.append(text)
                
                for i, text in enumerate(fee_items, 1):
                    response_text += f"**{i}.** {text}\n\n"
                
                if notes:
                    response_text += "\n**Important Notes:**\n"
                    for note in notes[:2]:
                        response_text += f"- {note}\n"
                    response_text += "\n"
                
                if contact:
                    response_text += "\n**Contact:**\n"
                    response_text += f"{contact[0]}\n\n"
            
            elif 'website' in detected_intents or 'portal' in detected_intents:
                # Website/portal queries
                response_text = "**Website Information:**\n\n"
                for i, result in enumerate(results[:5], 1):
                    response_text += f"**{i}.** {result['text']}\n\n"
            
            elif 'program' in detected_intents:
                # Program queries
                response_text = "**Programs at KARE:**\n\n"
                for i, result in enumerate(results[:8], 1):
                    response_text += f"**{i}.** {result['text']}\n\n"
            
            elif 'admission' in detected_intents:
                # Admission queries
                response_text = "**Admissions Information:**\n\n"
                for i, result in enumerate(results[:8], 1):
                    response_text += f"**{i}.** {result['text']}\n\n"
            
            elif 'placement' in detected_intents:
                # Placement queries
                response_text = "**Placements at KARE:**\n\n"
                for i, result in enumerate(results[:8], 1):
                    response_text += f"**{i}.** {result['text']}\n\n"
            
            elif 'contact' in detected_intents:
                # Contact queries
                response_text = "**Contact Information:**\n\n"
                for i, result in enumerate(results[:5], 1):
                    response_text += f"**{i}.** {result['text']}\n\n"
            
            else:
                # General queries - smart formatting
                response_text = "**Information from KARE:**\n\n"
                for i, result in enumerate(results[:6], 1):
                    response_text += f"**{i}.** {result['text']}\n\n"
            
            response_text += f"\n*(Source: {results[0]['source_file']})*"
            
            logger.info(f"Found {len(results)} results | Intents: {', '.join(detected_intents)}")
            return Response(response=response_text)
        else:
            return Response(
                response="I couldn't find information about that. Please try rephrasing your question."
            )
            
    except Exception as e:
        logger.error(f"Query error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
