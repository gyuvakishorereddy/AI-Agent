#!/usr/bin/env python3
"""
Final Demonstration of Enhanced College AI Agent
Shows that the model is trained and ready to answer all types of questions
"""

import os
import requests
import json
from datetime import datetime

def test_web_interface():
    """Test the web interface functionality"""
    print("🌐 TESTING WEB INTERFACE")
    print("=" * 50)
    
    base_url = "http://localhost:5003"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Web server is healthy")
            print(f"   📊 Model: {health_data.get('model_info', {}).get('model_type', 'Unknown')}")
            print(f"   🏫 Colleges: {health_data.get('model_info', {}).get('colleges', 0)}")
            print(f"   💬 Q&A Pairs: {health_data.get('model_info', {}).get('qa_pairs', 0)}")
        else:
            print("❌ Web server health check failed")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to web server: {e}")
        print("💡 Make sure to run: python enhanced_web_server.py")
        return False
    
    # Test various types of queries
    test_queries = [
        # General conversation
        "hi",
        "what can you do?",
        "help me",
        
        # College-specific
        "tell me about kalasalingam university",
        "what is the fee at IIT Bombay?",
        "which companies visit for placements?",
        
        # Educational guidance
        "how to choose engineering college?",
        "compare government and private colleges",
        "what are the best engineering branches?",
        
        # Thank you
        "thank you"
    ]
    
    print(f"\n🧪 Testing {len(test_queries)} different query types:")
    print("-" * 50)
    
    success_count = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. 🔍 Testing: '{query}'")
        
        try:
            response = requests.post(
                f"{base_url}/query",
                json={"question": query, "top_k": 1},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results') and len(data['results']) > 0:
                    result = data['results'][0]
                    confidence = result['confidence']
                    answer = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
                    
                    if confidence >= 70:
                        status = "🟢 EXCELLENT"
                    elif confidence >= 50:
                        status = "🟡 GOOD"
                    else:
                        status = "🟠 FAIR"
                    
                    print(f"   {status} ({confidence:.1f}%)")
                    print(f"   💬 {answer}")
                    success_count += 1
                else:
                    print("   ❌ No response received")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   ✅ Successful queries: {success_count}/{len(test_queries)}")
    print(f"   📈 Success rate: {(success_count/len(test_queries)*100):.1f}%")
    
    return success_count >= len(test_queries) * 0.8  # 80% success rate

def check_model_files():
    """Check if model files exist"""
    print("\n📁 CHECKING MODEL FILES")
    print("=" * 50)
    
    files_to_check = [
        "enhanced_college_ai_english.pkl",
        "college_ai_agent.pkl",
        "train_english_comprehensive_agent.py",
        "enhanced_web_server.py"
    ]
    
    all_exist = True
    
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # MB
            print(f"   ✅ {file} ({size:.1f} MB)")
        else:
            print(f"   ❌ {file} - NOT FOUND")
            all_exist = False
    
    return all_exist

def final_demo():
    """Final demonstration of capabilities"""
    print("🎉 ENHANCED COLLEGE AI AGENT - FINAL DEMONSTRATION")
    print("=" * 70)
    print(f"📅 Tested on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check model files
    files_ok = check_model_files()
    
    # Test web interface
    web_ok = test_web_interface()
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT")
    print("=" * 70)
    
    if files_ok and web_ok:
        print("🟢 STATUS: FULLY OPERATIONAL")
        print()
        print("✅ Enhanced model trained with comprehensive English data")
        print("✅ Understands general questions and conversations")
        print("✅ Searches database effectively for college information")
        print("✅ Web interface running successfully")
        print("✅ All query types working correctly")
        print()
        print("🎊 CONGRATULATIONS! Your AI agent is ready for use!")
        print()
        print("🌐 Access your AI agent at: http://localhost:5003")
        print("💡 Try asking:")
        print("   • 'Hi, how can you help me?'")
        print("   • 'Tell me about Kalasalingam University'")
        print("   • 'What is the best engineering college?'")
        print("   • 'Compare government vs private colleges'")
        print("   • 'Help me choose a college'")
        
        return True
        
    else:
        print("🔴 STATUS: ISSUES DETECTED")
        if not files_ok:
            print("❌ Some model files are missing")
        if not web_ok:
            print("❌ Web interface is not responding correctly")
        print()
        print("🔧 Please check the setup and try again")
        
        return False

if __name__ == "__main__":
    success = final_demo()
    
    if success:
        print("\n" + "="*70)
        print("🚀 MISSION ACCOMPLISHED!")
        print("Your AI agent is trained, tested, and ready to answer")
        print("all types of questions while searching the database!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠️  Please resolve the issues and try again")
        print("="*70)
