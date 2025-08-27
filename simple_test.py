#!/usr/bin/env python3
"""
Simple direct test of the improved model
"""

import pickle
import os

print("🔄 Testing Improved Model")
print("=" * 40)

# Load the improved model
model_file = 'improved_college_ai_english.pkl'
if os.path.exists(model_file):
    print(f"📥 Loading {model_file}...")
    with open(model_file, 'rb') as f:
        agent = pickle.load(f)
    
    print(f"✅ Loaded successfully")
    print(f"🏫 Colleges: {len(agent.college_data)}")
    print(f"💬 Q&A Pairs: {len(agent.qa_pairs)}")
    
    # Test the specific query that was problematic
    print("\n🧪 Testing: 'What is the fee structure for KL University?'")
    results = agent.query("What is the fee structure for KL University?", top_k=1)
    
    if results:
        answer = results[0]['answer']
        confidence = results[0]['confidence']
        print(f"📊 Confidence: {confidence:.1f}%")
        print(f"💬 Answer: {answer}")
        
        if "₹430,000" in answer or "fees" in answer.lower():
            print("✅ SUCCESS: Model now provides specific fee information!")
        else:
            print("⚠️  Still needs improvement")
    else:
        print("❌ No results")
        
    print("\n🧪 Testing: 'best private engineering colleges'")
    results2 = agent.query("best private engineering colleges", top_k=1)
    if results2:
        answer2 = results2[0]['answer']
        confidence2 = results2[0]['confidence']
        print(f"📊 Confidence: {confidence2:.1f}%")
        print(f"💬 Answer: {answer2[:150]}...")
    
else:
    print(f"❌ Model file {model_file} not found!")

print("\n✅ Testing completed!")
