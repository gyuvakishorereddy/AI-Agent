#!/usr/bin/env python3
"""
Test script to verify hostel booking queries work correctly
"""

import requests
import json
import time

# Wait for server to be ready
print("Waiting for server to be ready...")
time.sleep(3)

queries = [
    'what is the website for hostel booking',
    'how can i book hostel'
]

print("\n" + "=" * 80)
print("TESTING HOSTEL BOOKING QUERIES")
print("=" * 80)

for query in queries:
    print(f"\n\nQuery: {query}")
    print("-" * 80)
    
    try:
        response = requests.post(
            'http://localhost:8000/api/query',
            json={'query': query, 'language': 'en'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')
            print(f"RESPONSE:\n{response_text}")
            
            # Check if response contains hostel information
            if 'hostel' in response_text.lower() or 'booking' in response_text.lower():
                print("\n✅ SUCCESS: Response contains hostel information")
            else:
                print("\n⚠️  WARNING: Response may not contain expected hostel information")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
