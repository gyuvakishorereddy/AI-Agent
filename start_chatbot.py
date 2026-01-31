"""
KARE AI Chatbot Startup Script
Run this from the root directory: python start_chatbot.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Change to project root directory
os.chdir(project_root)

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print("=" * 80)
print("KARE AI CHATBOT - STARTING...")
print("=" * 80)
print(f"Working Directory: {os.getcwd()}")
print(f"Python Path: {sys.path[0]}")
print()

# Import and run the app
try:
    from src.app_simple import app
    import uvicorn
    
    print("Modules loaded successfully")
    print("Starting server on http://localhost:8002")
    print("=" * 80)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
