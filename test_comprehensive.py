#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.ERROR)
from src.rag_engine import RAGResponseGenerator

rag = RAGResponseGenerator(vector_store_path='faiss_index')

print("="*80)
print("TEST 1: DOCUMENTS REQUIRED")
print("="*80)
resp = rag.generate_response('what documents required for joining', language='en')
print(resp)

print("\n" + "="*80)
print("TEST 2: ELIGIBILITY CRITERIA")  
print("="*80)
resp2 = rag.generate_response('what is the eligibility criteria for admission', language='en')
print(resp2[:1500])
