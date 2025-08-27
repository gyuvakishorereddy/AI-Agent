#!/usr/bin/env python3
"""
Check what's in the pickle file
"""

import pickle
import os

print("🔍 Inspecting Improved Model File")
print("=" * 40)

model_file = 'improved_college_ai_english.pkl'
if os.path.exists(model_file):
    print(f"📥 Loading {model_file}...")
    with open(model_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Loaded successfully")
    print(f"📊 Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"🔑 Keys: {list(data.keys())}")
        for key, value in data.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {type(value)}")
    
    # Try to access qa_pairs directly if it's a dict
    if isinstance(data, dict) and 'qa_pairs' in data:
        print(f"\n💬 Q&A Pairs: {len(data['qa_pairs'])}")
        
        # Test a sample query manually
        print("\n🧪 Testing manual search...")
        query = "KL University fees"
        query_lower = query.lower()
        
        matches = []
        for qa in data['qa_pairs'][:100]:  # Check first 100
            question = qa['question'].lower()
            answer = qa['answer'].lower()
            if 'kl' in question or 'kalasalingam' in question:
                if 'fee' in question or 'fee' in answer:
                    matches.append({
                        'question': qa['question'],
                        'answer': qa['answer'][:200]
                    })
        
        print(f"🎯 Found {len(matches)} potential matches")
        for i, match in enumerate(matches[:3], 1):
            print(f"\n{i}. Q: {match['question']}")
            print(f"   A: {match['answer']}...")

else:
    print(f"❌ Model file {model_file} not found!")
