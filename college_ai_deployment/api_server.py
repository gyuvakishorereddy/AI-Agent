from flask import Flask, request, jsonify
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

# Load model on startup
print("Loading College AI Agent...")
with open('college_ai_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pairs = model_data['qa_pairs']
embeddings = model_data['embeddings']

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

@app.route('/query', methods=['POST'])
def query_agent():
    try:
        data = request.json
        question = data.get('question', '')
        top_k = data.get('top_k', 5)
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Create embedding
        question_embedding = sentence_model.encode([question])
        faiss.normalize_L2(question_embedding)
        
        # Search
        scores, indices = index.search(question_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(qa_pairs):
                qa = qa_pairs[idx]
                results.append({
                    'college': qa['college'],
                    'category': qa['category'],
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'confidence': float(score) * 100
                })
        
        return jsonify({
            'query': question,
            'results': results,
            'total_results': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'total_colleges': len(set(qa['college'] for qa in qa_pairs)),
        'total_qa_pairs': len(qa_pairs)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)