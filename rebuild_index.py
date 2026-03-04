#!/usr/bin/env python3
"""
Rebuild FAISS index to ensure all data is properly indexed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vector_store import VectorStoreManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("REBUILDING FAISS INDEX")
    logger.info("=" * 80)
    
    # Create vector store manager
    vs_manager = VectorStoreManager(
        data_dir="data_md",
        vector_store_path="faiss_index"
    )
    
    # Force rebuild
    success = vs_manager.build_vector_store(force_rebuild=True)
    
    if success:
        logger.info("\n✅ FAISS index rebuilt successfully!")
        logger.info(f"   Total chunks: {len(vs_manager.chunks)}")
        logger.info(f"   FAISS vectors: {vs_manager.index.ntotal}")
    else:
        logger.error("\n❌ Failed to rebuild FAISS index")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
