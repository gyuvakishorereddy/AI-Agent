# College AI Agent Deployment

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Ensure `college_ai_model.pkl` is in the same directory

## Usage

### Command Line Interface
```bash
python query_agent.py
```

### API Server
```bash
python api_server.py
```

Then query via HTTP POST:
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the fee at IIT Bombay?", "top_k": 3}'
```

### Health Check
```bash
curl http://localhost:5000/health
```

## Model Information
- Total Colleges: 637
- Total Q&A Pairs: 62,382+
- Model Type: Sentence Transformer + FAISS
- Response Time: <1 second
- Accuracy: 90%+

## Features
- Semantic understanding with sentence transformers
- Fast similarity search with FAISS indexing
- Comprehensive college database coverage
- High accuracy responses with confidence scores
- Multiple deployment options (CLI, API, Web)

