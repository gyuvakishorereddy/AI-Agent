#!/usr/bin/env python3
"""
Fixed College AI Agent Web Interface
Robust web server with better error handling
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import traceback

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Global agent variable
agent = None

def initialize_agent():
    """Initialize the AI agent with error handling"""
    global agent
    try:
        print("🚀 Initializing College AI Agent...")
        
        from train_college_ai_agent import CollegeAIAgent
        
        # Create agent
        agent = CollegeAIAgent(enable_multilingual=False)
        
        # Load the trained model
        print("📥 Loading trained model...")
        agent.load_model("college_ai_agent.pkl")
        
        print("✅ Agent initialized successfully!")
        print(f"📊 Loaded {len(agent.colleges_data)} colleges")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Main page with improved interface"""
    agent_status = "✅ Ready" if agent else "❌ Not Ready"
    college_count = len(agent.colleges_data) if agent else 0
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>College AI Agent - Fixed</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; text-align: center; }}
            .status {{ background: #e3f2fd; padding: 15px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #2196F3; }}
            .chat-box {{ border: 1px solid #ddd; height: 450px; overflow-y: auto; padding: 15px; margin: 15px 0; background: #fafafa; border-radius: 8px; }}
            .message {{ margin: 12px 0; padding: 12px; border-radius: 8px; }}
            .user {{ background: #e3f2fd; text-align: right; }}
            .bot {{ background: #f1f8e9; }}
            .error {{ background: #ffebee; color: #c62828; }}
            .input-group {{ display: flex; margin-top: 15px; }}
            input[type="text"] {{ flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 5px 0 0 5px; font-size: 14px; }}
            button {{ padding: 12px 24px; background: #4CAF50; color: white; border: none; border-radius: 0 5px 5px 0; cursor: pointer; font-size: 14px; }}
            button:hover {{ background: #45a049; }}
            .suggestions {{ background: #fff3e0; padding: 12px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ff9800; }}
            .suggestions ul {{ margin: 8px 0; padding-left: 20px; }}
            .loading {{ background: #fff; padding: 8px; border-radius: 4px; text-align: center; font-style: italic; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 College AI Agent - Fixed Version</h1>
            
            <div class="status">
                <strong>📊 System Status:</strong> {agent_status} | 
                <strong>🏫 Colleges:</strong> {college_count} | 
                <strong>📚 Database:</strong> {'Loaded' if agent else 'Loading...'} |
                <strong>🌍 Language:</strong> English
            </div>
            
            <div class="suggestions">
                <strong>💡 Try these questions:</strong>
                <ul>
                    <li><strong>College-specific:</strong> "Tell me about Kalasalingam University"</li>
                    <li><strong>General:</strong> "How to apply for admission in 2025?"</li>
                    <li><strong>Fees:</strong> "What are the fees at IIT Bombay?"</li>
                    <li><strong>Placements:</strong> "Tell me about placement statistics"</li>
                </ul>
            </div>
            
            <div class="chat-box" id="chatBox">
                <div class="message bot">
                    <strong>🤖 AI Agent:</strong> Hello! I'm now properly loaded with data from {college_count} engineering colleges. 
                    <br><br>I can help you with specific college information (like Kalasalingam University) or general guidance about admissions, fees, and placements.
                    <br><br><strong>What would you like to know?</strong>
                </div>
            </div>
            
            <div class="input-group">
                <input type="text" id="userInput" placeholder="Ask about engineering colleges (e.g., 'Tell me about Kalasalingam University')..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            function sendMessage() {{
                const input = document.getElementById('userInput');
                const chatBox = document.getElementById('chatBox');
                const question = input.value.trim();
                
                if (!question) return;
                
                // Add user message
                const userMsg = document.createElement('div');
                userMsg.className = 'message user';
                userMsg.innerHTML = '<strong>You:</strong> ' + question;
                chatBox.appendChild(userMsg);
                
                // Add loading message
                const loadingMsg = document.createElement('div');
                loadingMsg.className = 'message bot loading';
                loadingMsg.id = 'loading';
                loadingMsg.innerHTML = '<strong>🤖 AI Agent:</strong> Searching database... 🔍';
                chatBox.appendChild(loadingMsg);
                
                input.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;
                
                // Send request
                fetch('/query', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{question: question}})
                }})
                .then(response => response.json())
                .then(data => {{
                    // Remove loading message
                    const loading = document.getElementById('loading');
                    if (loading) loading.remove();
                    
                    const botMsg = document.createElement('div');
                    botMsg.className = 'message bot';
                    
                    if (data.success && data.results && data.results.length > 0) {{
                        const result = data.results[0];
                        botMsg.innerHTML = `
                            <strong>🤖 AI Agent:</strong><br>
                            <strong>🏫 College:</strong> ${{result.college}}<br>
                            <strong>📂 Category:</strong> ${{result.category}}<br>
                            <strong>🎯 Confidence:</strong> ${{result.confidence.toFixed(1)}}%<br>
                            <strong>💬 Answer:</strong> ${{result.answer}}
                        `;
                    }} else {{
                        botMsg.className = 'message error';
                        botMsg.innerHTML = `
                            <strong>❌ No Results Found</strong><br>
                            Error: ${{data.error || 'No matching information found'}}<br>
                            Total results: ${{data.total_results || 0}}<br>
                            <br><strong>Suggestions:</strong>
                            <ul>
                                <li>Try "Tell me about [College Name]"</li>
                                <li>Use full college names (e.g., "Kalasalingam University")</li>
                                <li>Ask general questions about admissions or fees</li>
                            </ul>
                        `;
                    }}
                    chatBox.appendChild(botMsg);
                    chatBox.scrollTop = chatBox.scrollHeight;
                }})
                .catch(error => {{
                    const loading = document.getElementById('loading');
                    if (loading) loading.remove();
                    
                    const errorMsg = document.createElement('div');
                    errorMsg.className = 'message error';
                    errorMsg.innerHTML = '<strong>🤖 System Error:</strong> ' + error.message;
                    chatBox.appendChild(errorMsg);
                    chatBox.scrollTop = chatBox.scrollHeight;
                }});
            }}
        </script>
    </body>
    </html>
    '''

@app.route('/query', methods=['POST'])
def query():
    """Handle user queries with detailed logging"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        print(f"\n📝 Received query: '{question}'")
        
        if not question:
            return jsonify({'success': False, 'error': 'No question provided'})
        
        if agent is None:
            return jsonify({'success': False, 'error': 'Agent not initialized'})
        
        # Get response from agent
        print(f"🔍 Processing query...")
        results = agent.query_agent(question, top_k=3)
        
        print(f"💬 Query results: {len(results) if results else 0} found")
        
        if results and len(results) > 0:
            print(f"   ✅ Best result: {results[0].get('college', 'Unknown')} - {results[0].get('confidence', 0):.1f}%")
            for i, result in enumerate(results[:3], 1):
                print(f"   {i}. {result.get('college', 'N/A')} ({result.get('confidence', 0):.1f}%)")
        else:
            print("   ❌ No results from main query")
            
            # Try fallback search
            print("🔍 Trying fallback search...")
            try:
                fallback_results = agent.fallback_search(question, top_k=3)
                if fallback_results:
                    print(f"   ✅ Fallback found {len(fallback_results)} results")
                    results = fallback_results
                else:
                    print("   ❌ No fallback results either")
            except Exception as fb_error:
                print(f"   ❌ Fallback error: {fb_error}")
        
        return jsonify({
            'success': True if results and len(results) > 0 else False,
            'results': results if results else [],
            'question': question,
            'total_results': len(results) if results else 0,
            'error': None if results and len(results) > 0 else 'No matching information found in database'
        })
        
    except Exception as e:
        print(f"❌ Error processing query '{question}': {e}")
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': str(e),
            'results': [],
            'total_results': 0
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if agent else 'initializing',
        'agent_loaded': agent is not None,
        'college_count': len(agent.colleges_data) if agent else 0
    })

@app.route('/colleges')
def list_colleges():
    """List all available colleges"""
    if agent:
        colleges = list(agent.colleges_data.keys())
        return jsonify({
            'success': True,
            'colleges': colleges[:50],  # First 50
            'total': len(colleges)
        })
    else:
        return jsonify({'success': False, 'error': 'Agent not loaded'})

if __name__ == '__main__':
    print("🚀 Starting Fixed College AI Agent Web Server...")
    print("=" * 60)
    
    if initialize_agent():
        print(f"🌐 Starting web server on http://localhost:5002")
        print(f"✅ Ready to answer questions about {len(agent.colleges_data)} colleges!")
        app.run(host='0.0.0.0', port=5002, debug=False, use_reloader=False)
    else:
        print("❌ Failed to start server - agent initialization failed")
