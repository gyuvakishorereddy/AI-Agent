#!/usr/bin/env python3
"""
Interactive AI Agent Runner
Test the AI agent with sample queries
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ai_agent():
    """Test the AI agent with sample queries"""
    try:
        from train_college_ai_agent import CollegeAIAgent
        
        print("🚀 Starting College AI Agent...")
        print("=" * 50)
        
        # Initialize agent
        agent = CollegeAIAgent(enable_multilingual=False)
        
        # Load the trained model
        print("📥 Loading trained model...")
        agent.load_model("college_ai_agent.pkl")
        print("✅ Model loaded successfully!")
        
        # Test queries
        test_queries = [
            "How to apply for admission in 2025?",
            "What are the fees at IIT Bombay?", 
            "Tell me about placement statistics",
            "What is the admission process for NIT Trichy?",
            "What are the facilities at VIT?",
            "How to get admission in engineering colleges?"
        ]
        
        print(f"\n🧪 Testing with {len(test_queries)} sample queries:")
        print("-" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            try:
                results = agent.query_agent(query, top_k=2)
                
                if results:
                    for j, result in enumerate(results, 1):
                        print(f"   Result {j}:")
                        print(f"     🏫 College: {result.get('college', 'N/A')}")
                        print(f"     📂 Category: {result.get('category', 'N/A')}")
                        print(f"     🎯 Confidence: {result.get('confidence', 0):.1f}%")
                        answer = result.get('answer', 'No answer available')
                        print(f"     💬 Answer: {answer[:200]}...")
                        print()
                else:
                    print("   ❌ No results returned")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n🎉 AI Agent testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ai_agent()
    if success:
        print("\n✅ AI Agent is running successfully!")
        print("🚀 Ready to answer queries about 637+ engineering colleges!")
    else:
        print("\n❌ AI Agent failed to start")
