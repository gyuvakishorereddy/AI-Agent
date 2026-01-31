"""
Build FAISS Vector Store from Markdown Files
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vector_store import VectorStoreManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

logger = logging.getLogger(__name__)

def main():
    logger.info("\n" + "="*80)
    logger.info("BUILDING FAISS VECTOR STORE FROM MARKDOWN FILES")
    logger.info("="*80 + "\n")
    
    # Initialize vector store manager
    manager = VectorStoreManager(
        data_dir="data_md",
        vector_store_path="faiss_index"
    )
    
    # Build vector store (force rebuild)
    success = manager.build_vector_store(force_rebuild=True)
    
    if success:
        logger.info("\n" + "="*80)
        logger.info("✅ SUCCESS - FAISS Vector Store Ready!")
        logger.info(f"📊 Total chunks: {len(manager.chunks)}")
        logger.info(f"📁 Stored in: faiss_index/")
        logger.info("="*80 + "\n")
    else:
        logger.error("\n" + "="*80)
        logger.error("❌ FAILED - Could not build vector store")
        logger.error("="*80 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
