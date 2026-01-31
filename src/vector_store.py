"""
FAISS Vector Store Manager
Loads Markdown files, chunks them, embeds them, and stores in FAISS
"""

import os
import json
import logging
import pickle
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manage FAISS vector store with markdown file chunking and embedding"""
    
    def __init__(self, data_dir: str = "data_md", vector_store_path: str = "faiss_index"):
        self.data_dir = data_dir
        self.vector_store_path = vector_store_path
        self.vector_store_file = os.path.join(vector_store_path, "faiss_index.bin")
        self.metadata_file = os.path.join(vector_store_path, "metadata.pkl")
        
        self.index = None
        self.chunks = []  # Store all chunks with metadata
        self.embeddings = []
        self.embedding_model = None
        
        # Create vector store directory
        os.makedirs(vector_store_path, exist_ok=True)
        
    def initialize_embedding_model(self):
        """Initialize sentence transformer for embeddings"""
        try:
            # Suppress TensorFlow warnings
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            
            import warnings
            warnings.filterwarnings('ignore')
            
            from sentence_transformers import SentenceTransformer
            logger.info("[LOAD] Loading embedding model: all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("[OK] Embedding model loaded successfully")
            return True
        except ImportError:
            logger.error("[ERROR] sentence-transformers not installed. Install: pip install sentence-transformers")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Failed to load embedding model: {e}")
            return False
    
    def chunk_markdown_file(self, filepath: str, chunk_size: int = 800) -> List[Dict[str, Any]]:
        """
        Load and chunk a single Markdown file
        Returns list of chunks with metadata
        """
        filename = os.path.basename(filepath)
        logger.info(f"📄 Processing: {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = []
            
            # Split by double newlines (paragraphs/sections)
            sections = content.split('\n\n')
            
            for idx, section in enumerate(sections):
                section = section.strip()
                if not section or len(section) < 20:
                    continue
                
                # If section is too large, split it further
                if len(section) > chunk_size:
                    sub_chunks = self._split_text(section, chunk_size)
                    for sub_idx, sub_chunk in enumerate(sub_chunks):
                        chunks.append({
                            'text': sub_chunk,
                            'source_file': filename.replace('.md', ''),
                            'section_index': idx,
                            'chunk_index': sub_idx,
                            'metadata': {}
                        })
                else:
                    chunks.append({
                        'text': section,
                        'source_file': filename.replace('.md', ''),
                        'section_index': idx,
                        'chunk_index': 0,
                        'metadata': {}
                    })
            
            logger.info(f"  ✅ Created {len(chunks)} chunks from {filename}")
            return chunks
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {filename}: {e}")
            return []
    
    def chunk_json_file(self, filepath: str, chunk_size: int = 500) -> List[Dict[str, Any]]:
        """
        Load and chunk a single JSON file
        Returns list of chunks with metadata
        """
        filename = os.path.basename(filepath)
        logger.info(f"📄 Processing: {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of items - chunk each item
                for idx, item in enumerate(data):
                    text = self._extract_text_from_item(item)
                    if len(text) > chunk_size:
                        # Split large items into smaller chunks
                        sub_chunks = self._split_text(text, chunk_size)
                        for sub_idx, sub_chunk in enumerate(sub_chunks):
                            chunks.append({
                                'text': sub_chunk,
                                'source_file': filename,
                                'item_index': idx,
                                'chunk_index': sub_idx,
                                'metadata': item if isinstance(item, dict) else {}
                            })
                    else:
                        chunks.append({
                            'text': text,
                            'source_file': filename,
                            'item_index': idx,
                            'chunk_index': 0,
                            'metadata': item if isinstance(item, dict) else {}
                        })
            
            elif isinstance(data, dict):
                # Single object - extract key sections
                for key, value in data.items():
                    text = self._extract_text_from_item({key: value})
                    if len(text) > chunk_size:
                        sub_chunks = self._split_text(text, chunk_size)
                        for sub_idx, sub_chunk in enumerate(sub_chunks):
                            chunks.append({
                                'text': sub_chunk,
                                'source_file': filename,
                                'key': key,
                                'chunk_index': sub_idx,
                                'metadata': {key: value}
                            })
                    else:
                        chunks.append({
                            'text': text,
                            'source_file': filename,
                            'key': key,
                            'chunk_index': 0,
                            'metadata': {key: value}
                        })
            
            logger.info(f"  ✅ Created {len(chunks)} chunks from {filename}")
            return chunks
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {filename}: {e}")
            return []
    
    def _extract_text_from_item(self, item: Any) -> str:
        """Extract searchable text from JSON item"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            text_parts = []
            for key, value in item.items():
                if isinstance(value, (str, int, float, bool)):
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, list):
                    text_parts.append(f"{key}: {' '.join(str(v) for v in value)}")
                elif isinstance(value, dict):
                    text_parts.append(self._extract_text_from_item(value))
            return " | ".join(text_parts)
        elif isinstance(item, list):
            return " | ".join(self._extract_text_from_item(i) for i in item)
        else:
            return str(item)
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately chunk_size characters"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def build_vector_store(self, force_rebuild: bool = False):
        """Build FAISS vector store from all JSON files in data directory"""
        
        # Check if vector store already exists
        if not force_rebuild and os.path.exists(self.vector_store_file):
            logger.info("📦 Loading existing vector store...")
            return self.load_vector_store()
        
        logger.info("="*80)
        logger.info("🔨 BUILDING FAISS VECTOR STORE")
        logger.info("="*80)
        
        # Initialize embedding model
        if not self.initialize_embedding_model():
            logger.error("❌ Cannot build vector store without embedding model")
            return False
        
        # Process all Markdown files
        all_chunks = []
        md_files = [f for f in os.listdir(self.data_dir) if f.endswith('.md')]
        
        logger.info(f"📁 Found {len(md_files)} Markdown files to process")
        print()
        
        for md_file in sorted(md_files):
            filepath = os.path.join(self.data_dir, md_file)
            file_chunks = self.chunk_markdown_file(filepath)
            all_chunks.extend(file_chunks)
        
        if not all_chunks:
            logger.error("❌ No chunks created from Markdown files")
            return False
        
        logger.info(f"\n📊 Total chunks created: {len(all_chunks)}")
        logger.info("🔄 Generating embeddings...")
        
        # Generate embeddings for all chunks
        try:
            texts = [chunk['text'] for chunk in all_chunks]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            
            # Build FAISS index
            try:
                import faiss
            except Exception as e:
                logger.error(f"Failed to import FAISS: {e}. Trying alternate import...")
                import faiss_cpu as faiss
            
            dimension = embeddings.shape[1]
            logger.info(f"🔧 Building FAISS index (dimension: {dimension})")
            
            # Use IndexFlatL2 for exact search (good for small-medium datasets)
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            
            logger.info(f"✅ FAISS index built with {self.index.ntotal} vectors")
            
            # Store chunks and save
            self.chunks = all_chunks
            self.embeddings = embeddings
            
            # Save vector store
            self.save_vector_store()
            
            logger.info("="*80)
            logger.info("✅ VECTOR STORE BUILD COMPLETE")
            logger.info("="*80)
            return True
            
        except ImportError:
            logger.error("❌ faiss-cpu not installed. Install: pip install faiss-cpu")
            return False
        except Exception as e:
            logger.error(f"❌ Error building vector store: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_vector_store(self):
        """Save FAISS index and metadata to disk"""
        try:
            try:
                import faiss
            except Exception:
                import faiss_cpu as faiss
            
            # Save FAISS index
            faiss.write_index(self.index, self.vector_store_file)
            logger.info(f"💾 Saved FAISS index to {self.vector_store_file}")
            
            # Save metadata (chunks without embeddings to save space)
            metadata = {
                'chunks': self.chunks,
                'total_vectors': self.index.ntotal
            }
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"💾 Saved metadata to {self.metadata_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving vector store: {e}")
    
    def load_vector_store(self):
        """Load FAISS index and metadata from disk"""
        try:
            try:
                import faiss
            except Exception:
                import faiss_cpu as faiss
            
            # Load FAISS index
            self.index = faiss.read_index(self.vector_store_file)
            logger.info(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load metadata
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            self.chunks = metadata['chunks']
            logger.info(f"✅ Loaded {len(self.chunks)} chunks metadata")
            
            # Note: Embedding model will be initialized lazily on first search
            # This avoids blocking app startup with model downloads
            logger.info("⏳ Embedding model will be loaded on first search")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading vector store: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search vector store for relevant chunks
        Returns list of top_k most similar chunks
        """
        if self.index is None:
            logger.warning("⚠️ Vector store not loaded")
            return []
        
        # Initialize embedding model on first search (lazy loading)
        if self.embedding_model is None:
            logger.info("🔄 Initializing embedding model for first search...")
            if not self.initialize_embedding_model():
                logger.error("❌ Failed to initialize embedding model")
                return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            
            # Search FAISS index
            distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Get corresponding chunks
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    chunk['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity
                    results.append(chunk)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching vector store: {e}")
            return []


# Global instance
vector_store_manager = VectorStoreManager()


def build_vector_store(force_rebuild: bool = False):
    """Utility function to build vector store"""
    return vector_store_manager.build_vector_store(force_rebuild)


def search_vector_store(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Utility function to search vector store"""
    return vector_store_manager.search(query, top_k)
