"""
FAISS Vector Store Manager
Loads Markdown files, chunks them, embeds them, and stores in FAISS
"""

import os
import json
import logging
import pickle
import re
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
    
    def chunk_markdown_file(self, filepath: str, chunk_size: int = 500) -> List[Dict[str, Any]]:
        """
        Load and chunk a single Markdown file
        Returns list of chunks with metadata  
        Uses section-aware chunking to preserve context
        """
        filename = os.path.basename(filepath)
        logger.info(f"📄 Processing: {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = []
            
            # Split by sections (## headers) for better context
            sections = re.split(r'\n(?=##\s)', content)
            
            for idx, section in enumerate(sections):
                section = section.strip()
                if not section or len(section) < 20:
                    continue
                
                # Extract section header for context
                header = ''
                header_match = re.match(r'^(#+\s+.+)', section)
                if header_match:
                    header = header_match.group(1).lstrip('#').strip()
                
                # Further split large sections into sub-sections (### headers)
                sub_sections = re.split(r'\n(?=###\s)', section)
                
                for sub_idx, sub_section in enumerate(sub_sections):
                    sub_section = sub_section.strip()
                    if not sub_section or len(sub_section) < 15:
                        continue
                    
                    # Extract sub-header
                    sub_header = ''
                    sub_header_match = re.match(r'^(#+\s+.+)', sub_section)
                    if sub_header_match:
                        sub_header = sub_header_match.group(1).lstrip('#').strip()
                    
                    # Add context prefix from headers
                    context_prefix = ''
                    if header:
                        context_prefix = f"Topic: {header}"
                        if sub_header and sub_header != header:
                            context_prefix += f" > {sub_header}"
                        context_prefix += "\n"
                    
                    # If sub-section is small enough, keep as one chunk
                    if len(sub_section) <= chunk_size:
                        chunk_text = context_prefix + sub_section
                        chunks.append({
                            'text': chunk_text,
                            'source_file': filename.replace('.md', ''),
                            'section_index': idx,
                            'chunk_index': sub_idx,
                            'header': header,
                            'sub_header': sub_header,
                            'metadata': {}
                        })
                    else:
                        # Split large sub-sections by paragraphs
                        paragraphs = sub_section.split('\n\n')
                        current_chunk = context_prefix
                        chunk_count = 0
                        
                        for para in paragraphs:
                            para = para.strip()
                            if not para:
                                continue
                            
                            if len(current_chunk) + len(para) > chunk_size and len(current_chunk) > len(context_prefix) + 10:
                                chunks.append({
                                    'text': current_chunk.strip(),
                                    'source_file': filename.replace('.md', ''),
                                    'section_index': idx,
                                    'chunk_index': chunk_count,
                                    'header': header,
                                    'sub_header': sub_header,
                                    'metadata': {}
                                })
                                chunk_count += 1
                                current_chunk = context_prefix + para + '\n'
                            else:
                                current_chunk += para + '\n'
                        
                        # Don't forget the last chunk
                        if len(current_chunk.strip()) > len(context_prefix) + 10:
                            chunks.append({
                                'text': current_chunk.strip(),
                                'source_file': filename.replace('.md', ''),
                                'section_index': idx,
                                'chunk_index': chunk_count,
                                'header': header,
                                'sub_header': sub_header,
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
        Uses hybrid approach: FAISS vector search + keyword boosting + source file boosting
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
            
            # Search FAISS index - retrieve many more candidates for re-ranking
            search_k = min(top_k * 8, self.index.ntotal)
            distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            # Extract query keywords for boosting
            query_lower = query.lower()
            query_words = set(query_lower.replace('?', ' ').replace('!', ' ').replace(',', ' ').split())
            # Remove stop words
            stop_words = {'what', 'is', 'the', 'for', 'a', 'an', 'of', 'in', 'to', 'and',
                         'how', 'where', 'when', 'which', 'who', 'can', 'do', 'does', 'are',
                         'was', 'were', 'be', 'been', 'about', 'with', 'from', 'at', 'by',
                         'this', 'that', 'i', 'me', 'my', 'tell', 'give', 'show', 'please',
                         'want', 'know', 'need', 'like', 'get', 'will', 'would', 'could',
                         'should', 'have', 'has', 'had', 'there', 'their', 'they', 'its'}
            keywords = query_words - stop_words
            
            # Special handling: if query asks for website + something, boost websites source heavily
            asks_for_website = any(w in query_lower for w in ['website', 'url', 'portal', 'link', 'online', 'booking'])
            
            # Map query to likely source files for source boosting
            source_hints = []
            source_map = {
                'hostel': ['hostels'], 'room': ['hostels'], 'warden': ['hostels'], 'fresher': ['hostels'],
                'fee': ['fees', 'hostels', 'scholarships'], 'fees': ['fees', 'hostels'],
                'cost': ['fees', 'hostels'], 'tuition': ['fees'], 'tariff': ['hostels'],
                'admission': ['admissions', 'websites'], 'apply': ['admissions', 'websites'], 'eligib': ['admissions'],
                'entrance': ['admissions'], 'enroll': ['admissions', 'websites'],
                'placement': ['placements'], 'recruit': ['placements'], 'package': ['placements'],
                'company': ['placements'], 'salary': ['placements'], 'lpa': ['placements'],
                'bus': ['transport'], 'transport': ['transport'], 'route': ['transport'],
                'fare': ['transport'], 'ticket': ['transport'],
                'mess': ['mess', 'hostels'], 'food': ['mess'], 'canteen': ['mess'],
                'breakfast': ['mess'], 'lunch': ['mess'], 'dinner': ['mess'],
                'scholarship': ['scholarships'], 'waiver': ['scholarships'],
                'loan': ['scholarships'], 'jee': ['scholarships'],
                'website': ['websites'], 'url': ['websites'], 'portal': ['websites'],
                'login': ['websites'], 'link': ['websites'], 'online': ['websites'],
                'booking': ['websites', 'hostels'], 'apply': ['websites', 'admissions'],
                'contact': ['contact'], 'phone': ['contact'], 'email': ['contact'],
                'number': ['contact'], 'helpline': ['contact'], 'toll': ['contact'],
                'address': ['contact'], 'location': ['contact'], 'reach': ['contact'],
                'department': ['departments'], 'faculty': ['departments'], 'hod': ['departments'],
                'facility': ['facilities'], 'library': ['facilities'], 'lab': ['facilities'],
                'sports': ['facilities'], 'gym': ['facilities'], 'medical': ['facilities'],
                'wifi': ['facilities'], 'swimming': ['facilities'], 'canteen': ['facilities'],
                'research': ['research'], 'patent': ['research'], 'innovation': ['research'],
                'incubat': ['research'], 'startup': ['research'], 'journal': ['research'],
                'program': ['programs'], 'course': ['programs'], 'degree': ['programs'], 'offer': ['programs'],
                'btech': ['programs', 'fees'], 'mtech': ['programs', 'fees'],
                'mba': ['programs', 'fees'], 'mca': ['programs', 'fees'],
                'phd': ['programs', 'research', 'fees'], 'msc': ['programs', 'fees'],
                'engineering': ['programs', 'departments'], 'management': ['programs'],
                'club': ['student_life'], 'event': ['student_life'], 'fest': ['student_life'],
                'ncc': ['student_life'], 'nss': ['student_life'], 'activity': ['student_life'],
                'block': ['academic_blocks'], 'building': ['academic_blocks'],
                'classroom': ['academic_blocks'], 'campus': ['academic_blocks'],
                'naac': ['programs'], 'nba': ['programs'], 'accredit': ['programs'],
            }
            for word in keywords:
                for key, sources in source_map.items():
                    if key in word:
                        source_hints.extend(sources)
            source_hints = list(set(source_hints))
            
            # If query asks for website and no specific source hints yet, prioritize websites
            if asks_for_website and 'websites' not in source_hints:
                source_hints = ['websites'] + source_hints
            
            # Score and rank candidates
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    # Convert L2 distance to cosine similarity
                    cosine_sim = float(max(0.0, 1.0 - distance / 2.0))
                    
                    # Keyword boost: add up to 0.20 for keyword matches
                    chunk_text_lower = chunk.get('text', '').lower()
                    keyword_matches = sum(1 for kw in keywords if kw in chunk_text_lower)
                    keyword_boost = min(0.20, keyword_matches * 0.04)
                    
                    # Source file boost: strong boost if chunk comes from expected source
                    source_file = chunk.get('source_file', '').lower()
                    source_match = any(s in source_file for s in source_hints)
                    source_boost = 0.25 if source_match else -0.05  # Penalize non-matching sources
                    
                    # Header match bonus: if chunk header contains query keywords
                    chunk_header = chunk.get('header', '').lower()
                    chunk_sub_header = chunk.get('sub_header', '').lower()
                    header_keywords = sum(1 for kw in keywords if kw in chunk_header or kw in chunk_sub_header)
                    header_boost = min(0.15, header_keywords * 0.05)
                    
                    # Combined score
                    final_score = max(0.0, cosine_sim + keyword_boost + source_boost + header_boost)
                    
                    chunk['similarity_score'] = final_score
                    chunk['cosine_score'] = cosine_sim
                    chunk['keyword_boost'] = keyword_boost
                    chunk['source_boost'] = source_boost
                    chunk['header_boost'] = header_boost
                    chunk['l2_distance'] = float(distance)
                    results.append(chunk)
            
            # Sort by final score descending and return top_k
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:top_k]
            
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
