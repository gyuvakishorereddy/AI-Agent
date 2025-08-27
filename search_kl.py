#!/usr/bin/env python3
"""
Search for KL University data in the model
"""

import pickle
import os

print("🔍 Searching for KL University Data")
print("=" * 40)

model_file = 'improved_college_ai_english.pkl'
with open(model_file, 'rb') as f:
    data = pickle.load(f)

print(f"💬 Total Q&A Pairs: {len(data['qa_pairs'])}")

# Search for KL University related entries
kl_matches = []
kalasalingam_matches = []

for i, qa in enumerate(data['qa_pairs']):
    question = qa['question'].lower()
    answer = qa['answer'].lower()
    
    # Look for various KL University references
    if any(term in question for term in ['kl ', 'kalasalingam']):
        if 'fee' in question or 'cost' in question:
            kl_matches.append({
                'index': i,
                'question': qa['question'],
                'answer': qa['answer'][:300]
            })
    
    # Also check college name variations
    if 'kalasalingam' in question or 'klef' in question:
        kalasalingam_matches.append({
            'index': i,
            'question': qa['question'],
            'answer': qa['answer'][:200]
        })

print(f"\n🎯 Found {len(kl_matches)} fee-related KL matches")
for i, match in enumerate(kl_matches[:5], 1):
    print(f"\n{i}. Q: {match['question']}")
    print(f"   A: {match['answer']}")

print(f"\n🏫 Found {len(kalasalingam_matches)} Kalasalingam matches")
for i, match in enumerate(kalasalingam_matches[:3], 1):
    print(f"\n{i}. Q: {match['question']}")
    print(f"   A: {match['answer']}")

# Check college name variations
print(f"\n🔍 College name variations:")
if 'college_name_variations' in data:
    variations = data['college_name_variations']
    kl_variations = {k: v for k, v in variations.items() if 'kl' in k.lower() or 'kalasalingam' in k.lower()}
    for key, value in kl_variations.items():
        print(f"  {key}: {value}")

# Check if we have specific college data
print(f"\n📊 Colleges data available: {len(data['colleges_data'])}")
kl_colleges = {k: v for k, v in data['colleges_data'].items() if 'kalasalingam' in k.lower() or 'kl' in k.lower()}
print(f"🎯 KL related colleges: {len(kl_colleges)}")
for name in kl_colleges.keys():
    print(f"  - {name}")
