#!/usr/bin/env python3
"""
Test RAG engine for debugging specific queries
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_engine import RAGResponseGenerator
from vector_store import VectorStoreManager
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test queries
test_queries = [
    ("what is the website for hostel booking", "en"),
    ("what is the website for admission application", "en"),
    ("சேர்க்கை விண்ணப்பத்திற்கான வலைத்தளம் என்ன", "ta"),  # Tamil: What is the website for admission application
]

def main():
    logger.info("=" * 80)
    logger.info("KARE RAG Test - Diagnosing Query Issues")
    logger.info("=" * 80)
    
    # Initialize RAG engine
    rag_engine = RAGResponseGenerator(
        vector_store_path="faiss_index",
        data_dir="data_md",
        use_llm=False
    )
    
    if not rag_engine.vector_store or not rag_engine.vector_store.index:
        logger.error("FAISS index not loaded!")
        return
    
    logger.info(f"\n✅ FAISS Index loaded with {len(rag_engine.vector_store.chunks)} chunks\n")
    
    # Test each query
    for query, lang in test_queries:
        logger.info("=" * 80)
        logger.info(f"Query: '{query}'")
        logger.info(f"Language: {lang}")
        logger.info("-" * 80)
        
        # Perform search
        chunks = rag_engine.vector_store.search(query, top_k=10)
        
        if chunks:
            logger.info(f"Found {len(chunks)} relevant chunks:")
            for i, chunk in enumerate(chunks[:5]):  # Show top 5
                logger.info(f"\n  [{i+1}] Score: {chunk.get('similarity_score', 0):.3f} | "
                          f"Source: {chunk.get('source_file', 'unknown')}")
                logger.info(f"      Text preview: {chunk.get('text', '')[:150]}...")
        else:
            logger.warning("No chunks found!")
        
        # Generate response
        response = rag_engine.generate_response(query, language=lang, top_k=5)
        logger.info(f"\n📝 Response:\n{response}\n")

if __name__ == "__main__":
    main()
