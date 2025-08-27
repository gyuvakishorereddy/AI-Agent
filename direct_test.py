#!/usr/bin/env python3
"""
Direct test of the improved model without web interface
"""

import pickle
import os
from datetime import datetime

def load_and_test_model():
    print("🔄 Testing Improved Model Directly")
    print("=" * 50)
    
    # Load the improved model
    model_file = 'improved_college_ai_english.pkl'
    if not os.path.exists(model_file):
        print(f"❌ Model file {model_file} not found!")
        return
    
    print(f"📥 Loading {model_file}...")
    with open(model_file, 'rb') as f:
        agent = pickle.load(f)
    
    print(f"✅ Loaded successfully")
    print(f"🏫 Colleges: {len(agent.college_data)}")
    print(f"💬 Q&A Pairs: {len(agent.qa_pairs)}")
    print()
    
    # Test queries that were problematic before
    test_queries = [
        "What is the fee structure for KL University?",
        "Tell me about Kalasalingam University fees",
        "Which are the best private engineering colleges?",
        "What is the ranking of VIT University?",
        "KL university placement statistics"
    ]
    
    print("🧪 Testing improved responses:")
    print("-" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        try:
            results = agent.query(query, top_k=1)
            if results:
                answer = results[0]['answer']
                confidence = results[0]['confidence']
                print(f"   📊 Confidence: {confidence:.1f}%")
                print(f"   💬 Answer: {answer[:200]}...")
                if confidence > 120:
                    print("   ✅ High confidence - specific data found!")
                elif confidence > 100:
                    print("   🔸 Good confidence - relevant information")
                else:
                    print("   ⚠️  Lower confidence - may be generic")
            else:
                print("   ❌ No results returned")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Direct testing completed!")

if __name__ == "__main__":
    load_and_test_model()
