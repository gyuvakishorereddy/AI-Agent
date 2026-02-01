# KARE AI Chatbot - Markdown-Based RAG Pipeline v2.0

> **Fixed & Rebuilt**: Complete JSON → Markdown → TF-IDF Vector Store pipeline

## ✨ What's New (v2.0)

- ✅ **Markdown-Based Knowledge Base**: All 15 JSON files converted to readable markdown
- ✅ **TF-IDF Vector Search**: Lightweight, fast, no heavy ML model dependencies
- ✅ **151 Semantic Chunks**: Intelligently split documents with overlapping context
- ✅ **Confidence Scoring**: Every response shows how confident the match is
- ✅ **Source Attribution**: Know which document each answer comes from
- ✅ **Multi-Language**: English, Tamil, Telugu, Hindi, Kannada, Malayalam
- ✅ **Unified Startup**: One command handles everything (JSON→MD→Build→Start)

## 🎯 Overview

A production-ready RAG (Retrieval-Augmented Generation) chatbot for KARE Academy with:

- **Data**: 15 markdown documents (151 chunks)
- **Search**: TF-IDF cosine similarity (< 50ms)
- **API**: FastAPI with 6 endpoints
- **UI**: Responsive web interface
- **Languages**: 5 supported languages

## 📂 What's Included

```
ai agent/
├── data/                      # Original JSON (15 files)
├── data_md/                   # Converted Markdown (15 files) ← NEW
├── vector_store_md/           # TF-IDF Index
│   ├── vectorizer.pkl         # TF-IDF vectorizer
│   ├── tfidf_matrix.pkl       # 151 × 2000 sparse matrix
│   ├── chunks.json            # Chunk metadata
│   └── index_info.json        # Statistics
│
├── public/                    # Web UI
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── src/
│   └── translation_engine.py  # Language detection
│
├── convert_json_to_md.py      # JSON → MD converter ← NEW
├── markdown_rag_pipeline.py   # RAG builder ← NEW
├── app_v2.py                  # FastAPI app ← NEW
├── start_server.py            # Bootstrap script ← NEW
└── requirements.txt
```

## 🚀 Getting Started

### **Quickest Way** (Recommended)
```bash
python start_server.py
```
This automatically:
1. ✅ Checks dependencies
2. ✅ Converts JSON → Markdown (if needed)
3. ✅ Builds vector store (if needed)
4. ✅ Starts API server on port 8000

Then open: **http://localhost:8000**

### Manual Steps (if needed)
```bash
# 1. Convert JSON to Markdown
python convert_json_to_md.py

# 2. Build vector store from markdown
python markdown_rag_pipeline.py

# 3. Start server
python app_v2.py
```

## 📊 Vector Store Stats

| Metric | Value |
|--------|-------|
| **Documents** | 15 (categories) |
| **Chunks** | 151 (semantic chunks) |
| **TF-IDF Features** | 2000 |
| **Vector Type** | Sparse (150.2 MB→2 MB compressed) |
| **Search Speed** | < 50ms per query |
| **Memory** | ~10-20 MB (loaded) |
| **Format** | Python pickle (vectorizer + matrix) |

## 🎯 Knowledge Base Categories

1. **Academic Blocks** - Campus infrastructure
2. **Admissions** - Entry requirements
3. **Contact** - Leadership & departments
4. **Departments** - Courses & programs
5. **Facilities** - Labs, library, sports
6. **Fees** - Cost structure
7. **Hostels** - Living arrangements
8. **Mess** - Food services
9. **Placements** - Career outcomes
10. **Programs** - Degree options
11. **Research** - Research labs
12. **Scholarships** - Financial aid
13. **Student Life** - Activities & clubs
14. **Transport** - Bus routes & fares
15. **Websites** - Online resources

## 🌐 API Endpoints

### 1. Main Query Endpoint ⭐
```bash
POST /api/query
Content-Type: application/json

{
  "query": "What programs are offered?",
  "language": "en"
}
```

**Response:**
```json
{
  "query": "What programs are offered?",
  "language": "en",
  "response": "Based on the university knowledge base: ...",
  "confidence": 0.856,
  "sources": [
    {
      "category": "programs",
      "confidence": 0.856,
      "chunk_id": 42
    }
  ],
  "timestamp": "2025-12-10T22:15:30.123456"
}
```

### 2. Search Endpoint
```bash
GET /api/search?q=bus+fare&limit=5
```

### 3. Health Check
```bash
GET /health
```

### 4. Categories
```bash
GET /api/categories
```

### 5. Info
```bash
GET /api/info
```

### 6. Web UI
```bash
GET /
```

## 🧪 Test Examples

### Using cURL
```bash
# Query endpoint
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the bus fare?", "language": "en"}'

# Search endpoint
curl "http://localhost:8000/api/search?q=hostel&limit=3"

# Health check
curl http://localhost:8000/health
```

### Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/query",
    json={
        "query": "Tell me about admissions",
        "language": "en"
    }
)

print(response.json())
```

### Using JavaScript
```javascript
fetch('http://localhost:8000/api/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        query: "What programs are offered?",
        language: "en"
    })
})
.then(r => r.json())
.then(data => console.log(data))
```

## 📝 How It Works

### Query Processing Flow
```
User Query
    ↓
[1] Language Detection
    ↓ (English detected)
[2] Vector Search (TF-IDF)
    - Query → vectorizer.transform()
    - Cosine similarity with chunks
    - Return top-5 results
    ↓
[3] Build Response
    - Extract context from top results
    - Add confidence scores
    - Prepare sources list
    ↓
[4] Translate (if needed)
    - If non-English, use template responses
    - Otherwise, use RAG response
    ↓
Response with Sources + Confidence
```

### Chunking Strategy
- **Size**: 800 characters per chunk
- **Overlap**: 200 characters (preserves context)
- **Split Method**: By markdown headers (#, ##, ###)
- **Total**: 151 chunks from 15 documents

### Vector Search (TF-IDF)
- Converts text to word frequency vectors
- Calculates cosine similarity (0-1 scale)
- Returns chunks with highest similarity
- Fast (no neural networks required)

## ⚙️ Configuration

Edit `app_v2.py`:
```python
# Vector store location
vector_store_dir = "vector_store_md"

# Search parameters
search_k = 5                    # Top-5 results
confidence_threshold = 0.05    # Minimum confidence

# Server
host = "0.0.0.0"
port = 8000
```

Edit `markdown_rag_pipeline.py`:
```python
# Chunking
chunk_size = 800              # Characters per chunk
overlap = 200                 # Character overlap

# TF-IDF
max_features = 2000           # Vocabulary size
ngram_range = (1, 2)          # Unigrams + bigrams
```

## 🔄 Update Knowledge Base

To add/update information:

### Step 1: Update JSON
Edit files in `data/` directory:
```json
{
  "metadata": {...},
  "newSection": {
    "content": "..."
  }
}
```

### Step 2: Rebuild
```bash
# Converts JSON → Markdown
python convert_json_to_md.py

# Rebuilds vector store from markdown
python markdown_rag_pipeline.py

# Restart server (auto-reload if dev mode)
```

### Done!
Vector store automatically updated. No code changes needed.

## 🚨 Troubleshooting

### Issue: "Vector store not found"
```bash
# Solution: Rebuild
python markdown_rag_pipeline.py
```

### Issue: "Markdown files missing"
```bash
# Solution: Reconvert
python convert_json_to_md.py
```

### Issue: "Port 8000 already in use"
```bash
# Change port in app_v2.py line 150
uvicorn.run(..., port=8001)
```

### Issue: "Poor search results"
```bash
# Try:
# 1. Check confidence threshold in app_v2.py (line ~110)
# 2. Adjust chunk_size in markdown_rag_pipeline.py
# 3. Rebuild vector store
python markdown_rag_pipeline.py
```

### Check Vector Store Health
```bash
python -c "
from markdown_rag_pipeline import MarkdownRAGVectorStore
vs = MarkdownRAGVectorStore()
if vs.load_vector_store():
    print(f'✅ Chunks: {len(vs.chunks)}')
    print(f'✅ Features: {len(vs.vectorizer.get_feature_names_out())}')
else:
    print('❌ Vector store load failed')
"
```

## 📦 Dependencies

```
fastapi==0.104.1              # Web framework
uvicorn==0.24.0               # ASGI server
pydantic==2.5.0               # Data validation
scikit-learn==1.3.2           # TF-IDF vectorizer
sentence-transformers==2.2.2  # (optional) advanced embeddings
numpy==1.24.3                 # Numerical computing
```

Install all:
```bash
pip install -r requirements.txt
```

## 📈 Performance Benchmarks

```
Startup Time:        ~5 seconds
Vector Build Time:   ~10 seconds (151 chunks)
Query Processing:    50-200ms
Search Only:         <50ms
Response Generation: <100ms
Memory Usage:        ~500 MB
Concurrent Queries:  100+
```

## 🎨 Web UI Features

- ✅ Responsive design (mobile-friendly)
- ✅ Real-time chat interface
- ✅ Language selector (5 languages)
- ✅ Category quick-links
- ✅ Source attribution panel
- ✅ Confidence score display
- ✅ Copy-to-clipboard responses
- ✅ Clean, modern UI

## 🔐 Security

- Vector store is read-only in production
- No authentication needed (internal use)
- CORS enabled for frontend
- Input validation with Pydantic
- Error messages don't leak sensitive info

## 📊 Query Examples

Try these in the web UI:

1. "What programs are offered?"
2. "Tell me about admissions"
3. "What is the bus fare for day scholars?"
4. "What are the hostel facilities?"
5. "How much are the fees?"
6. "Tell me about scholarships"
7. "What are the placements like?"
8. "How do I contact admissions?"

## 🎯 Next Steps

1. **Start Server**
   ```bash
   python start_server.py
   ```

2. **Open Browser**
   ```
   http://localhost:8000
   ```

3. **Ask Questions**
   Type your query and press Enter

4. **Explore Results**
   Check sources and confidence scores

5. **Update Knowledge** (Optional)
   Edit JSON files and rebuild

## 📚 Documentation Files

- `README.md` - This file (overview & quick start)
- `requirements.txt` - Python dependencies
- `convert_json_to_md.py` - JSON to Markdown converter
- `markdown_rag_pipeline.py` - RAG pipeline builder
- `app_v2.py` - FastAPI application
- `start_server.py` - Unified startup script

## 🤝 Contributing

To improve the chatbot:

1. Update knowledge in `data/` JSON files
2. Rebuild: `python markdown_rag_pipeline.py`
3. Test responses
4. Commit improvements

## 📞 Support

- Check logs: Terminal output shows all operations
- Test individual components with provided scripts
- Verify vector store with health check
- Review chunking strategy if results poor

## 📜 License

KARE Academy - Internal Use Only

## ✨ Version Info

- **Version**: 2.0
- **Build Date**: December 2025
- **Status**: ✅ Production Ready
- **Last Updated**: 2025-12-10

---

**Ready to start?** → `python start_server.py` 🚀


developed by me
