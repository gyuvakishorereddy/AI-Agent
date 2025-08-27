#!/usr/bin/env python3
"""
Simple College AI Agent Web Interface
Quick web server for immediate use while multilingual server is starting
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Global agent variable
agent = None

def initialize_agent():
    """Initialize the AI agent"""
    global agent
    try:
        from train_college_ai_agent import CollegeAIAgent
        
        print("🚀 Initializing Simple College AI Agent...")
        agent = CollegeAIAgent(enable_multilingual=False)
        agent.load_model("college_ai_agent.pkl")
        print("✅ Agent initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>College AI Agent</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .chat-box { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 10px; margin: 10px 0; background: #fafafa; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user { background: #e3f2fd; text-align: right; }
            .bot { background: #f1f8e9; }
            .input-group { display: flex; margin-top: 10px; }
            input[type="text"] { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px 0 0 5px; }
            button { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 0 5px 5px 0; cursor: pointer; }
            button:hover { background: #45a049; }
            .stats { background: #e8f5e8; padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 College AI Agent</h1>
            <div class="stats">
                <strong>📊 System Status:</strong> ✅ Running | 
                <strong>🏫 Colleges:</strong> 637+ | 
                <strong>📚 Q&A Pairs:</strong> 61,745+ |
                <strong>🌍 Languages:</strong> English (Multilingual server starting...)
            </div>
            
            <div class="chat-box" id="chatBox">
                <div class="message bot">
                    <strong>🤖 AI Agent:</strong> Hello! I can help you with information about 637+ engineering colleges in India. 
                    <br><br>
                    <strong>Try asking:</strong>
                    <ul>
                        <li>"How to apply for admission in 2025?"</li>
                        <li>"What are the fees at IIT Bombay?"</li>
                        <li>"Tell me about placement statistics"</li>
                        <li>"What is the admission process for NIT Trichy?"</li>
                    </ul>
                </div>
            </div>
            
            <div class="input-group">
                <input type="text" id="userInput" placeholder="Ask about engineering colleges..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            function sendMessage() {
                const input = document.getElementById('userInput');
                const chatBox = document.getElementById('chatBox');
                const question = input.value.trim();
                
                if (!question) return;
                
                // Add user message
                const userMsg = document.createElement('div');
                userMsg.className = 'message user';
                userMsg.innerHTML = '<strong>You:</strong> ' + question;
                chatBox.appendChild(userMsg);
                
                // Add thinking message
                const thinkingMsg = document.createElement('div');
                thinkingMsg.className = 'message bot';
                thinkingMsg.id = 'thinking';
                thinkingMsg.innerHTML = '<strong>🤖 AI Agent:</strong> Thinking... 🤔';
                chatBox.appendChild(thinkingMsg);
                
                input.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;
                
                // Send request
                fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                })
                .then(response => response.json())
                .then(data => {
                    // Remove thinking message
                    document.getElementById('thinking').remove();
                    
                    if (data.success && data.results.length > 0) {
                        const result = data.results[0];
                        const botMsg = document.createElement('div');
                        botMsg.className = 'message bot';
                        botMsg.innerHTML = `
                            <strong>🤖 AI Agent:</strong><br>
                            <strong>🏫 College:</strong> ${result.college}<br>
                            <strong>📂 Category:</strong> ${result.category}<br>
                            <strong>🎯 Confidence:</strong> ${result.confidence.toFixed(1)}%<br>
                            <strong>💬 Answer:</strong> ${result.answer}
                        `;
                        chatBox.appendChild(botMsg);
                    } else {
                        const errorMsg = document.createElement('div');
                        errorMsg.className = 'message bot';
                        errorMsg.innerHTML = '<strong>🤖 AI Agent:</strong> Sorry, I couldn\\'t find relevant information. Please try rephrasing your question.';
                        chatBox.appendChild(errorMsg);
                    }
                    chatBox.scrollTop = chatBox.scrollHeight;
                })
                .catch(error => {
                    document.getElementById('thinking').remove();
                    const errorMsg = document.createElement('div');
                    errorMsg.className = 'message bot';
                    errorMsg.innerHTML = '<strong>🤖 AI Agent:</strong> ❌ Error: ' + error.message;
                    chatBox.appendChild(errorMsg);
                    chatBox.scrollTop = chatBox.scrollHeight;
                });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/query', methods=['POST'])
def query():
    """Handle user queries"""
    try:
        data = request.json
        question = data.get('question', '')
        
        print(f"📝 Received query: '{question}'")
        
        if not question:
            return jsonify({'success': False, 'error': 'No question provided'})
        
        if agent is None:
            return jsonify({'success': False, 'error': 'Agent not initialized'})
        
        # Get response from agent
        results = agent.query_agent(question, top_k=3)
        
        print(f"💬 Results: {len(results)} found")
        if results:
            print(f"   Best result: {results[0].get('college', 'Unknown')} - {results[0].get('confidence', 0):.1f}%")
        
        # If no results, try a broader search
        if not results or len(results) == 0:
            print("🔍 No results found, trying fallback search...")
            results = agent.fallback_search(question, top_k=3)
        
        return jsonify({
            'success': True if results and len(results) > 0 else False,
            'results': results if results else [],
            'question': question,
            'total_results': len(results) if results else 0
        })
        
    except Exception as e:
        print(f"❌ Error processing query '{question}': {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if agent else 'initializing',
        'agent_loaded': agent is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Simple College AI Agent Web Server...")
    print("=" * 50)
    
    if initialize_agent():
        print("🌐 Starting web server on http://localhost:5001")
        print("💡 Use this while the multilingual server (port 5000) is starting up")
        app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    else:
        print("❌ Failed to start server")
