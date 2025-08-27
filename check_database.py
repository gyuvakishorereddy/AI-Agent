#!/usr/bin/env python3
"""
Quick Database Check - See what colleges are available
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_colleges():
    """Check what colleges are in our database"""
    try:
        from train_college_ai_agent import CollegeAIAgent
        
        print("🔍 Checking College Database...")
        print("=" * 50)
        
        agent = CollegeAIAgent(enable_multilingual=False)
        agent.load_model("college_ai_agent.pkl")
        
        colleges = list(agent.colleges_data.keys())
        
        print(f"📊 Total colleges: {len(colleges)}")
        
        # Search for Kalasalingam
        kalasa_colleges = [c for c in colleges if 'kalasa' in c.lower()]
        print(f"\n🏫 Kalasalingam colleges found: {len(kalasa_colleges)}")
        for college in kalasa_colleges:
            print(f"   • {college}")
        
        # Show first 20 colleges as sample
        print(f"\n📋 Sample colleges (first 20):")
        for i, college in enumerate(colleges[:20], 1):
            print(f"   {i:2d}. {college}")
        
        # Test a query
        print(f"\n🧪 Testing query: 'kalasalingam university'")
        results = agent.query_agent("kalasalingam university", top_k=5)
        
        if results:
            print(f"✅ Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result.get('college', 'Unknown')} - {result.get('confidence', 0):.1f}%")
        else:
            print("❌ No results found")
            
            # Try fallback search
            print("\n🔍 Trying fallback search...")
            fallback_results = agent.fallback_search("kalasalingam", top_k=5)
            if fallback_results:
                print(f"✅ Fallback found {len(fallback_results)} results:")
                for i, result in enumerate(fallback_results, 1):
                    print(f"   {i}. {result.get('college', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_colleges()
