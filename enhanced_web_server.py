#!/usr/bin/env python3
"""
Enhanced Web Server for College AI Agent
Uses the comprehensive English model with improved general question handling
"""

from flask import Flask, request, jsonify, render_template_string
import pickle
import os
import logging
from datetime import datetime
from typing import Dict, List

# Import our enhanced agent
try:
    from train_english_comprehensive_agent import EnhancedCollegeAIAgent
    print("✅ Enhanced College AI Agent imported successfully")
except ImportError as e:
    print(f"❌ Error importing enhanced agent: {e}")
    # Fallback to basic loading
    EnhancedCollegeAIAgent = None

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent instance
agent = None
model_info = {}

def initialize_agent():
    """Initialize the enhanced agent"""
    global agent, model_info
    
    try:
        logger.info("🤖 Initializing Enhanced College AI Agent...")
        
        # Check if enhanced model exists
        enhanced_model_path = "enhanced_college_ai_english.pkl"
        fallback_model_path = "college_ai_agent.pkl"
        
        if os.path.exists(enhanced_model_path):
            logger.info("🎯 Loading enhanced English model...")
            if EnhancedCollegeAIAgent:
                agent = EnhancedCollegeAIAgent()
                if agent.load_model(enhanced_model_path):
                    model_info = {
                        'model_type': 'enhanced_english',
                        'model_path': enhanced_model_path,
                        'colleges': len(agent.colleges_data),
                        'qa_pairs': len(agent.qa_pairs),
                        'features': ['General Conversation', 'College Search', 'Career Guidance', 'Comparative Analysis']
                    }
                    logger.info("✅ Enhanced model loaded successfully")
                    return True
        
        # Fallback to regular model
        if os.path.exists(fallback_model_path):
            logger.info("📥 Loading fallback model...")
            with open(fallback_model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create a simple agent wrapper
            class SimpleAgent:
                def __init__(self, model_data):
                    from sentence_transformers import SentenceTransformer
                    import faiss
                    
                    self.qa_pairs = model_data['qa_pairs']
                    self.embeddings = model_data['embeddings']
                    self.colleges_data = model_data.get('colleges_data', {})
                    self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    # Create FAISS index
                    dimension = self.embeddings.shape[1]
                    self.index = faiss.IndexFlatIP(dimension)
                    self.index.add(self.embeddings)
                
                def query_agent(self, question, top_k=5):
                    import faiss
                    import numpy as np
                    
                    # Create question embedding
                    question_embedding = self.sentence_model.encode([question])
                    faiss.normalize_L2(question_embedding)
                    
                    # Search
                    scores, indices = self.index.search(question_embedding, top_k)
                    
                    results = []
                    for score, idx in zip(scores[0], indices[0]):
                        if idx < len(self.qa_pairs):
                            qa = self.qa_pairs[idx]
                            results.append({
                                'college': qa.get('college', 'Unknown'),
                                'category': qa.get('category', 'general'),
                                'question': qa['question'],
                                'answer': qa['answer'],
                                'confidence': float(score) * 100
                            })
                    
                    return results
            
            agent = SimpleAgent(model_data)
            model_info = {
                'model_type': 'basic',
                'model_path': fallback_model_path,
                'colleges': len(agent.colleges_data),
                'qa_pairs': len(agent.qa_pairs),
                'features': ['College Search', 'Basic Q&A']
            }
            logger.info("✅ Fallback model loaded successfully")
            return True
        
        logger.error("❌ No model files found")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error initializing agent: {e}")
        return False

@app.route('/')
def home():
    """Enhanced home page with better interface"""
    
    # Get model information
    model_status = "🟢 Online" if agent else "🔴 Offline"
    
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎓 Enhanced College AI Assistant</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                max-width: 800px;
                width: 100%;
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 300;
            }
            
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .status-bar {
                background: #f8f9fa;
                padding: 15px 30px;
                border-bottom: 1px solid #e9ecef;
            }
            
            .status-item {
                display: inline-block;
                margin-right: 20px;
                font-size: 0.9em;
                color: #666;
            }
            
            .chat-container {
                height: 400px;
                overflow-y: auto;
                padding: 20px 30px;
                background: #f8f9fa;
            }
            
            .message {
                margin-bottom: 15px;
                padding: 12px 16px;
                border-radius: 18px;
                max-width: 80%;
                word-wrap: break-word;
            }
            
            .user-message {
                background: #007bff;
                color: white;
                margin-left: auto;
                text-align: right;
            }
            
            .bot-message {
                background: white;
                color: #333;
                border: 1px solid #e9ecef;
                margin-right: auto;
            }
            
            .bot-message .confidence {
                font-size: 0.8em;
                color: #666;
                margin-top: 5px;
            }
            
            .input-container {
                padding: 20px 30px;
                background: white;
                border-top: 1px solid #e9ecef;
            }
            
            .input-group {
                display: flex;
                gap: 10px;
            }
            
            #userInput {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e9ecef;
                border-radius: 25px;
                font-size: 16px;
                outline: none;
                transition: border-color 0.3s;
            }
            
            #userInput:focus {
                border-color: #007bff;
            }
            
            #sendBtn {
                padding: 12px 24px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                transition: background 0.3s;
            }
            
            #sendBtn:hover {
                background: #0056b3;
            }
            
            #sendBtn:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            .suggestions {
                padding: 15px 30px;
                background: #f8f9fa;
                border-top: 1px solid #e9ecef;
            }
            
            .suggestion-chip {
                display: inline-block;
                background: white;
                color: #007bff;
                padding: 8px 16px;
                margin: 5px;
                border-radius: 20px;
                cursor: pointer;
                font-size: 0.9em;
                border: 1px solid #007bff;
                transition: all 0.3s;
            }
            
            .suggestion-chip:hover {
                background: #007bff;
                color: white;
            }
            
            .welcome-message {
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                padding: 20px;
                margin: 20px;
                border-radius: 15px;
                text-align: center;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
                color: #666;
            }
            
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #007bff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎓 Enhanced College AI Assistant</h1>
                <p>Your comprehensive guide to engineering colleges in India</p>
            </div>
            
            <div class="status-bar">
                <span class="status-item"><strong>Status:</strong> {{ model_status }}</span>
                <span class="status-item"><strong>Model:</strong> {{ model_info.model_type|title }}</span>
                <span class="status-item"><strong>Colleges:</strong> {{ model_info.colleges }}</span>
                <span class="status-item"><strong>Q&A Pairs:</strong> {{ model_info.qa_pairs }}</span>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="welcome-message">
                    <h3>👋 Welcome to Enhanced College AI!</h3>
                    <p>I can help you with detailed information about engineering colleges, admissions, fees, placements, and career guidance. Try asking me anything!</p>
                    <br>
                    <p><strong>Available Features:</strong></p>
                    <p>{{ model_info.features|join(' • ') }}</p>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <span>AI is thinking...</span>
            </div>
            
            <div class="suggestions">
                <strong>💡 Try these questions:</strong><br>
                <span class="suggestion-chip" onclick="askQuestion('Hi, how can you help me?')">👋 Hello</span>
                <span class="suggestion-chip" onclick="askQuestion('Tell me about Kalasalingam University')">🏫 Kalasalingam University</span>
                <span class="suggestion-chip" onclick="askQuestion('What is the fee structure at IIT Bombay?')">💰 IIT Bombay Fees</span>
                <span class="suggestion-chip" onclick="askQuestion('Which companies visit for placements?')">💼 Placements</span>
                <span class="suggestion-chip" onclick="askQuestion('How to get admission in engineering colleges?')">📝 Admissions</span>
                <span class="suggestion-chip" onclick="askQuestion('Compare government and private colleges')">⚖️ Compare Colleges</span>
            </div>
            
            <div class="input-container">
                <div class="input-group">
                    <input type="text" id="userInput" placeholder="Ask me anything about engineering colleges..." 
                           onkeypress="handleKeyPress(event)">
                    <button id="sendBtn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>

        <script>
            function addMessage(message, isUser, confidence = null) {
                const chatContainer = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
                
                let content = message;
                if (!isUser && confidence !== null) {
                    content += `<div class="confidence">Confidence: ${confidence.toFixed(1)}%</div>`;
                }
                
                messageDiv.innerHTML = content;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            function showLoading(show) {
                document.getElementById('loading').style.display = show ? 'block' : 'none';
                document.getElementById('sendBtn').disabled = show;
            }
            
            async function sendMessage() {
                const input = document.getElementById('userInput');
                const question = input.value.trim();
                
                if (!question) return;
                
                // Add user message
                addMessage(question, true);
                input.value = '';
                
                // Show loading
                showLoading(true);
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: question, top_k: 1 })
                    });
                    
                    const data = await response.json();
                    
                    if (data.results && data.results.length > 0) {
                        const result = data.results[0];
                        addMessage(result.answer, false, result.confidence);
                    } else {
                        addMessage("I apologize, but I couldn't find a relevant answer. Please try rephrasing your question or ask about specific engineering colleges.", false);
                    }
                } catch (error) {
                    addMessage("Sorry, there was an error processing your request. Please try again.", false);
                    console.error('Error:', error);
                } finally {
                    showLoading(false);
                }
            }
            
            function askQuestion(question) {
                document.getElementById('userInput').value = question;
                sendMessage();
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            // Add welcome message on load
            window.onload = function() {
                console.log('🤖 Enhanced College AI Assistant loaded successfully');
            };
        </script>
    </body>
    </html>
    '''
    
    return render_template_string(html_template, 
                                  model_status=model_status, 
                                  model_info=model_info)

@app.route('/query', methods=['POST'])
def query_agent():
    """Handle enhanced queries with comprehensive responses"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        top_k = data.get('top_k', 3)
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        if not agent:
            return jsonify({'error': 'AI agent not initialized'}), 500
        
        logger.info(f"🔍 Processing query: '{question}'")
        
        # Query the agent
        results = agent.query_agent(question, top_k=top_k)
        
        if results:
            logger.info(f"✅ Found {len(results)} results, best confidence: {results[0]['confidence']:.1f}%")
        else:
            logger.warning("⚠️ No results found")
        
        return jsonify({
            'query': question,
            'results': results,
            'total_results': len(results),
            'model_type': model_info.get('model_type', 'unknown'),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error processing query: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    return jsonify({
        'status': 'healthy' if agent else 'error',
        'model_loaded': agent is not None,
        'model_info': model_info,
        'timestamp': datetime.now().isoformat(),
        'server_type': 'enhanced_college_ai'
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get comprehensive statistics"""
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        # Calculate statistics
        stats = {
            'total_colleges': len(getattr(agent, 'colleges_data', {})),
            'total_qa_pairs': len(getattr(agent, 'qa_pairs', [])),
            'model_info': model_info,
            'capabilities': {
                'general_conversation': True,
                'college_search': True,
                'career_guidance': True,
                'comparative_analysis': True,
                'multilingual': False  # This is the English-only model
            }
        }
        
        # Add category breakdown if available
        if hasattr(agent, 'qa_pairs'):
            categories = {}
            for qa in agent.qa_pairs:
                category = qa.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
            stats['categories'] = categories
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': f'Error getting stats: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced College AI Agent Web Server")
    print("=" * 60)
    
    # Initialize the agent
    if initialize_agent():
        print(f"✅ Agent initialized successfully")
        print(f"📊 Model: {model_info['model_type']}")
        print(f"🏫 Colleges: {model_info['colleges']}")
        print(f"💬 Q&A Pairs: {model_info['qa_pairs']}")
        print(f"🎯 Features: {', '.join(model_info['features'])}")
        print("\n🌐 Starting web server on http://localhost:5003")
        print("💡 Try asking: 'Hi', 'Tell me about Kalasalingam University', 'Help me choose a college'")
        
        app.run(host='0.0.0.0', port=5003, debug=False)
    else:
        print("❌ Failed to initialize agent")
        print("🔧 Please ensure model files are available")
