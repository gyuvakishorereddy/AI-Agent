#!/usr/bin/env python3
"""
Comprehensive English Training for College AI Agent
Enhanced with general question understanding and improved database search
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

# Core ML libraries
from sentence_transformers import SentenceTransformer
import faiss

# For general knowledge and conversation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Try to download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

class EnhancedCollegeAIAgent:
    """Enhanced College AI Agent with comprehensive English training and general Q&A"""
    
    def __init__(self, data_path: str = "college_data"):
        print("🤖 Initializing Enhanced College AI Agent...")
        
        self.data_path = Path(data_path)
        self.colleges_data = {}
        self.qa_pairs = []
        self.embeddings = None
        self.index = None
        
        # Initialize NLP tools
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            
        # General knowledge responses
        self.general_responses = self._initialize_general_responses()
        
        # Load data
        self.load_college_data()
        self.prepare_comprehensive_training_data()
        
    def _initialize_general_responses(self) -> Dict[str, List[str]]:
        """Initialize general response patterns"""
        return {
            'greeting': [
                "Hello! I'm your College AI Assistant. I can help you with information about engineering colleges, admissions, fees, placements, and much more. What would you like to know?",
                "Hi there! I'm here to help you with college-related information. Feel free to ask about any engineering college, admission process, or career guidance.",
                "Welcome! I can assist you with detailed information about engineering colleges across India. How can I help you today?"
            ],
            'introduction': [
                "I'm an AI assistant specialized in providing comprehensive information about engineering colleges. I have data on over 600+ engineering colleges including IITs, NITs, and private institutions.",
                "I'm your personal guide for engineering college information. I can help with admissions, fees, placements, facilities, courses, and much more across 600+ colleges."
            ],
            'help': [
                "I can help you with:\n• College information (fees, courses, facilities)\n• Admission processes and eligibility\n• Placement statistics and companies\n• Rankings and comparisons\n• Career guidance\n• Entrance exam information\n\nJust ask me anything about engineering colleges!",
                "Here's what I can assist with:\n✅ College details and rankings\n✅ Fee structures and scholarships\n✅ Admission procedures\n✅ Placement records\n✅ Course information\n✅ Campus facilities\n✅ Entrance exams\n\nWhat specific information do you need?"
            ],
            'thank_you': [
                "You're welcome! I'm glad I could help. Feel free to ask if you have any more questions about colleges or admissions.",
                "Happy to help! If you need any more information about engineering colleges, just let me know.",
                "You're most welcome! I'm here whenever you need assistance with college-related queries."
            ],
            'goodbye': [
                "Goodbye! Best of luck with your college search and admissions. Feel free to come back anytime for more help!",
                "Take care! I hope the information was helpful. Don't hesitate to ask if you need more guidance in the future.",
                "Farewell! Wishing you success in your engineering journey. See you next time!"
            ]
        }
    
    def load_college_data(self):
        """Load all college data from JSON files"""
        print("📚 Loading college data...")
        
        if not self.data_path.exists():
            print(f"❌ Data path {self.data_path} not found")
            return
        
        college_count = 0
        for college_dir in self.data_path.iterdir():
            if college_dir.is_dir():
                college_data = {}
                college_name = college_dir.name
                
                # Load each JSON file
                json_files = [
                    'basic_info.json', 'courses.json', 'fees_structure.json',
                    'admission_process.json', 'facilities.json', 'placements.json', 'faq.json'
                ]
                
                for json_file in json_files:
                    file_path = college_dir / json_file
                    if file_path.exists():
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                college_data[json_file.replace('.json', '')] = json.load(f)
                        except Exception as e:
                            print(f"⚠️ Error loading {file_path}: {e}")
                
                if college_data:
                    self.colleges_data[college_name] = college_data
                    college_count += 1
        
        print(f"✅ Loaded {college_count} colleges")
    
    def prepare_comprehensive_training_data(self):
        """Prepare comprehensive Q&A pairs with enhanced general knowledge"""
        print("🎓 Preparing comprehensive training data...")
        
        self.qa_pairs = []
        
        # Add general conversation pairs
        self._add_general_conversation_pairs()
        
        # Add college-specific data
        for college_name, college_data in self.colleges_data.items():
            self._generate_college_qa_pairs(college_name, college_data)
        
        # Add comparative and analytical questions
        self._add_comparative_questions()
        
        # Add guidance and advice questions
        self._add_guidance_questions()
        
        print(f"✅ Generated {len(self.qa_pairs)} comprehensive Q&A pairs")
    
    def _add_general_conversation_pairs(self):
        """Add general conversation and greeting pairs"""
        
        # Greetings
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'namaste']
        for greeting in greetings:
            self.qa_pairs.append({
                'question': greeting,
                'answer': self.general_responses['greeting'][0],
                'college': 'General',
                'category': 'conversation',
                'confidence_boost': 0.2
            })
        
        # Introduction questions
        intro_questions = [
            'who are you', 'what can you do', 'what is your purpose', 'tell me about yourself',
            'what information do you have', 'how can you help me'
        ]
        for question in intro_questions:
            self.qa_pairs.append({
                'question': question,
                'answer': self.general_responses['introduction'][0],
                'college': 'General',
                'category': 'introduction',
                'confidence_boost': 0.2
            })
        
        # Help questions
        help_questions = [
            'help', 'what can you tell me', 'what information do you provide',
            'how to use this', 'guide me', 'what should I ask'
        ]
        for question in help_questions:
            self.qa_pairs.append({
                'question': question,
                'answer': self.general_responses['help'][0],
                'college': 'General',
                'category': 'help',
                'confidence_boost': 0.2
            })
        
        # Thank you responses
        thanks = ['thank you', 'thanks', 'appreciate it', 'grateful']
        for thank in thanks:
            self.qa_pairs.append({
                'question': thank,
                'answer': self.general_responses['thank_you'][0],
                'college': 'General',
                'category': 'gratitude',
                'confidence_boost': 0.2
            })
        
        # Goodbye responses
        goodbyes = ['bye', 'goodbye', 'see you', 'take care']
        for goodbye in goodbyes:
            self.qa_pairs.append({
                'question': goodbye,
                'answer': self.general_responses['goodbye'][0],
                'college': 'General',
                'category': 'farewell',
                'confidence_boost': 0.2
            })
    
    def _generate_college_qa_pairs(self, college_name: str, college_data: Dict):
        """Generate comprehensive Q&A pairs for a college"""
        
        # Basic information questions
        if 'basic_info' in college_data:
            basic_info = college_data['basic_info']
            
            # College overview
            if 'about' in basic_info:
                self.qa_pairs.extend([
                    {
                        'question': f'Tell me about {college_name}',
                        'answer': f"{college_name} is {basic_info.get('about', 'a prestigious engineering institution')}",
                        'college': college_name,
                        'category': 'basic_info'
                    },
                    {
                        'question': f'What is {college_name}',
                        'answer': f"{college_name} is {basic_info.get('about', 'an engineering college')}",
                        'college': college_name,
                        'category': 'basic_info'
                    }
                ])
            
            # Location
            if 'location' in basic_info:
                location = basic_info['location']
                location_text = ""
                if isinstance(location, dict):
                    city = location.get('city', '')
                    state = location.get('state', '')
                    location_text = f"{city}, {state}" if city and state else str(location)
                else:
                    location_text = str(location)
                
                if location_text:
                    self.qa_pairs.append({
                        'question': f'Where is {college_name} located',
                        'answer': f"{college_name} is located in {location_text}",
                        'college': college_name,
                        'category': 'location'
                    })
            
            # Ranking
            if 'ranking' in basic_info:
                ranking = basic_info['ranking']
                if isinstance(ranking, dict) and 'nirf_ranking' in ranking:
                    self.qa_pairs.append({
                        'question': f'What is the NIRF ranking of {college_name}',
                        'answer': f"The NIRF ranking of {college_name} is {ranking['nirf_ranking']}",
                        'college': college_name,
                        'category': 'ranking'
                    })
        
        # Fees questions
        if 'fees_structure' in college_data:
            fees = college_data['fees_structure']
            if isinstance(fees, dict) and 'undergraduate' in fees:
                ug_fees = fees['undergraduate']
                if isinstance(ug_fees, dict):
                    for course, fee_info in ug_fees.items():
                        try:
                            if isinstance(fee_info, dict) and 'total_fees' in fee_info:
                                self.qa_pairs.extend([
                                    {
                                        'question': f'What is the fee for {course} at {college_name}',
                                        'answer': f"The total fee for {course} at {college_name} is {fee_info['total_fees']}",
                                        'college': college_name,
                                        'category': 'fees'
                                    },
                                    {
                                        'question': f'How much does {course} cost at {college_name}',
                                        'answer': f"The {course} program at {college_name} costs {fee_info['total_fees']}",
                                        'college': college_name,
                                        'category': 'fees'
                                    }
                                ])
                        except Exception as e:
                            continue  # Skip problematic fee entries
        
        # Placement questions
        if 'placements' in college_data:
            placements = college_data['placements']
            
            if isinstance(placements, dict):
                if 'statistics' in placements:
                    stats = placements['statistics']
                    if isinstance(stats, dict) and 'average_package' in stats:
                        self.qa_pairs.extend([
                            {
                                'question': f'What is the average package at {college_name}',
                                'answer': f"The average placement package at {college_name} is {stats['average_package']}",
                                'college': college_name,
                                'category': 'placements'
                            },
                            {
                                'question': f'What are the placement statistics of {college_name}',
                                'answer': f"At {college_name}, the average package is {stats['average_package']} with {stats.get('placement_percentage', 'high')} placement rate",
                                'college': college_name,
                                'category': 'placements'
                            }
                        ])
                
                if 'top_recruiters' in placements:
                    companies = placements['top_recruiters']
                    if companies and isinstance(companies, list):
                        companies_list = ', '.join(companies[:5]) if len(companies) > 5 else ', '.join(companies)
                        self.qa_pairs.append({
                            'question': f'Which companies visit {college_name} for placements',
                            'answer': f"Top companies that recruit from {college_name} include {companies_list}",
                            'college': college_name,
                            'category': 'placements'
                        })
        
        # Admission questions
        if 'admission_process' in college_data:
            admission = college_data['admission_process']
            
            if isinstance(admission, dict) and 'entrance_exams' in admission:
                exams = admission['entrance_exams']
                if exams and isinstance(exams, list):
                    exams_list = ', '.join(exams)
                    self.qa_pairs.extend([
                        {
                            'question': f'How to get admission in {college_name}',
                            'answer': f"Admission to {college_name} is through {exams_list}",
                            'college': college_name,
                            'category': 'admission'
                        },
                        {
                            'question': f'What entrance exams are required for {college_name}',
                            'answer': f"For admission to {college_name}, you need to appear for {exams_list}",
                            'college': college_name,
                            'category': 'admission'
                        }
                    ])
        
        # Courses questions
        if 'courses' in college_data:
            courses = college_data['courses']
            
            if isinstance(courses, dict) and 'undergraduate_programs' in courses:
                ug_programs = courses['undergraduate_programs']
                if isinstance(ug_programs, dict) and 'engineering' in ug_programs:
                    engineering = ug_programs['engineering']
                    if isinstance(engineering, dict) and 'btech' in engineering:
                        btech = engineering['btech']
                        if isinstance(btech, dict) and 'departments' in btech:
                            departments = btech['departments']
                            if isinstance(departments, list):
                                dept_names = []
                                for dept in departments:
                                    if isinstance(dept, dict) and 'name' in dept:
                                        dept_names.append(dept['name'])
                                
                                if dept_names:
                                    dept_list = ', '.join(dept_names[:5]) if len(dept_names) > 5 else ', '.join(dept_names)
                                    
                                    self.qa_pairs.extend([
                                        {
                                            'question': f'What courses are offered at {college_name}',
                                            'answer': f"{college_name} offers various engineering courses including {dept_list}",
                                            'college': college_name,
                                            'category': 'courses'
                                        },
                                        {
                                            'question': f'What branches are available at {college_name}',
                                            'answer': f"Available engineering branches at {college_name} include {dept_list}",
                                            'college': college_name,
                                            'category': 'courses'
                                        }
                                    ])
        
        # Facilities questions
        if 'facilities' in college_data:
            facilities = college_data['facilities']
            if isinstance(facilities, dict):
                facility_names = list(facilities.keys())
                if facility_names:
                    facility_list = ', '.join(facility_names)
                    self.qa_pairs.append({
                        'question': f'What facilities are available at {college_name}',
                        'answer': f"{college_name} provides excellent facilities including {facility_list}",
                        'college': college_name,
                        'category': 'facilities'
                    })
    
    def _add_comparative_questions(self):
        """Add comparative and analytical questions"""
        
        # General comparison questions
        comparative_questions = [
            {
                'question': 'Compare IIT and NIT colleges',
                'answer': 'IITs are premier autonomous institutions with higher rankings and more research focus, while NITs are also excellent institutions with strong industry connections. Both offer quality engineering education with good placements.',
                'college': 'Comparative',
                'category': 'comparison'
            },
            {
                'question': 'Which are the best engineering colleges in India',
                'answer': 'The top engineering colleges in India include IIT Bombay, IIT Delhi, IIT Madras, IIT Kanpur, IIT Kharagpur, NIT Trichy, BITS Pilani, and other premier institutions with excellent academics and placements.',
                'college': 'General',
                'category': 'ranking'
            },
            {
                'question': 'What is the difference between government and private colleges',
                'answer': 'Government colleges typically have lower fees and established reputation, while private colleges often have modern infrastructure, industry partnerships, and flexible curricula. Both can offer quality education.',
                'college': 'General',
                'category': 'comparison'
            }
        ]
        
        self.qa_pairs.extend(comparative_questions)
    
    def _add_guidance_questions(self):
        """Add career guidance and advice questions"""
        
        guidance_questions = [
            {
                'question': 'How to choose the right engineering college',
                'answer': 'Consider factors like NIRF ranking, placement records, faculty quality, infrastructure, location, fees, and your career goals. Research thoroughly and visit campuses if possible.',
                'college': 'General',
                'category': 'guidance'
            },
            {
                'question': 'What are the best engineering branches',
                'answer': 'Popular engineering branches include Computer Science, Electronics, Mechanical, Civil, and newer fields like AI/ML, Data Science, and Robotics. Choose based on your interests and market demand.',
                'college': 'General',
                'category': 'guidance'
            },
            {
                'question': 'How to prepare for engineering entrance exams',
                'answer': 'Focus on strong fundamentals in Physics, Chemistry, and Mathematics. Practice regularly, take mock tests, solve previous year papers, and maintain a consistent study schedule.',
                'college': 'General',
                'category': 'guidance'
            },
            {
                'question': 'What is the admission process for engineering colleges',
                'answer': 'Most engineering admissions are through entrance exams like JEE Main, JEE Advanced, state CETs, or university-specific exams. Some colleges also offer management quota and NRI quota admissions.',
                'college': 'General',
                'category': 'admission'
            },
            {
                'question': 'When do engineering admissions start',
                'answer': 'Engineering admissions typically start in May-June after Class 12 results. JEE Main is conducted multiple times a year, and counseling processes run from June to September.',
                'college': 'General',
                'category': 'admission'
            }
        ]
        
        self.qa_pairs.extend(guidance_questions)
    
    def create_embeddings(self):
        """Create embeddings for all Q&A pairs"""
        print("🧠 Creating embeddings...")
        
        if not self.qa_pairs:
            print("❌ No Q&A pairs found")
            return
        
        # Create embeddings for questions
        questions = [qa['question'] for qa in self.qa_pairs]
        
        print(f"🔄 Processing {len(questions)} questions...")
        self.embeddings = self.sentence_model.encode(questions, show_progress_bar=True)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"✅ Created embeddings with dimension {dimension}")
    
    def classify_query(self, question: str) -> Dict[str, any]:
        """Enhanced query classification"""
        question_lower = question.lower().strip()
        
        # Simple greeting detection
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in question_lower for greeting in greetings):
            return {
                'type': 'greeting',
                'confidence': 0.9,
                'requires_search': False
            }
        
        # Help/guidance detection
        help_keywords = ['help', 'guide', 'what can you', 'how to', 'advice']
        if any(keyword in question_lower for keyword in help_keywords):
            return {
                'type': 'guidance',
                'confidence': 0.8,
                'requires_search': True
            }
        
        # College-specific detection
        college_indicators = ['college', 'university', 'iit', 'nit', 'iiit', 'bits']
        has_college_name = any(college in question_lower for college in self.colleges_data.keys())
        has_college_indicator = any(indicator in question_lower for indicator in college_indicators)
        
        if has_college_name or has_college_indicator:
            return {
                'type': 'college_specific',
                'confidence': 0.9,
                'requires_search': True
            }
        
        # General educational query
        education_keywords = ['admission', 'fee', 'placement', 'course', 'branch', 'engineering', 'exam']
        if any(keyword in question_lower for keyword in education_keywords):
            return {
                'type': 'educational',
                'confidence': 0.8,
                'requires_search': True
            }
        
        return {
            'type': 'general',
            'confidence': 0.6,
            'requires_search': True
        }
    
    def enhanced_search(self, question: str, top_k: int = 5) -> List[Dict]:
        """Enhanced search with better matching"""
        if not self.index or not self.qa_pairs:
            return []
        
        # Create question embedding
        question_embedding = self.sentence_model.encode([question])
        faiss.normalize_L2(question_embedding)
        
        # Search for similar questions
        scores, indices = self.index.search(question_embedding, min(top_k * 2, len(self.qa_pairs)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.qa_pairs):
                qa = self.qa_pairs[idx]
                confidence = float(score) * 100
                
                # Apply confidence boost for general conversation
                if qa.get('confidence_boost'):
                    confidence += qa['confidence_boost'] * 100
                
                results.append({
                    'college': qa['college'],
                    'category': qa['category'],
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'confidence': confidence
                })
        
        # Filter and sort results
        results = [r for r in results if r['confidence'] > 30]  # Lower threshold for general questions
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results[:top_k]
    
    def query_agent(self, question: str, top_k: int = 5) -> List[Dict]:
        """Enhanced query processing with intelligent responses"""
        
        # Classify the query
        classification = self.classify_query(question)
        
        # Handle greetings directly
        if classification['type'] == 'greeting':
            return [{
                'college': 'AI Assistant',
                'category': 'conversation',
                'question': question,
                'answer': self.general_responses['greeting'][0],
                'confidence': 95.0
            }]
        
        # Search for answers
        if classification['requires_search']:
            results = self.enhanced_search(question, top_k)
            
            # If no good results, provide helpful guidance
            if not results or (results and results[0]['confidence'] < 50):
                guidance_answer = """I understand you're looking for information about engineering colleges. I can help you with:

• **College Information**: Details about 600+ engineering colleges
• **Admissions**: Entrance exams, eligibility, application process
• **Fees**: Complete fee structure for different courses
• **Placements**: Company visits, packages, placement statistics
• **Courses**: Available branches and specializations
• **Facilities**: Infrastructure, hostels, labs, libraries

Please ask me specific questions like:
- "Tell me about IIT Bombay"
- "What is the fee for CSE at VIT?"
- "Which companies visit for placements at NIT Trichy?"
- "How to get admission in engineering colleges?"

How can I help you today?"""
                
                return [{
                    'college': 'AI Assistant',
                    'category': 'guidance',
                    'question': question,
                    'answer': guidance_answer,
                    'confidence': 75.0
                }]
            
            return results
        
        return []
    
    def save_model(self, model_path: str = "enhanced_college_ai_english.pkl"):
        """Save the enhanced English model"""
        print(f"💾 Saving enhanced model to {model_path}...")
        
        model_data = {
            'qa_pairs': self.qa_pairs,
            'embeddings': self.embeddings,
            'colleges_data': self.colleges_data,
            'general_responses': self.general_responses,
            'model_info': {
                'creation_date': datetime.now().isoformat(),
                'total_colleges': len(self.colleges_data),
                'total_qa_pairs': len(self.qa_pairs),
                'model_type': 'enhanced_english',
                'sentence_transformer': 'all-MiniLM-L6-v2'
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model saved successfully ({file_size:.1f} MB)")
    
    def load_model(self, model_path: str = "enhanced_college_ai_english.pkl"):
        """Load a pre-trained enhanced model"""
        print(f"📥 Loading enhanced model from {model_path}...")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.qa_pairs = model_data['qa_pairs']
            self.embeddings = model_data['embeddings']
            self.colleges_data = model_data['colleges_data']
            self.general_responses = model_data.get('general_responses', self._initialize_general_responses())
            
            # Recreate FAISS index
            if self.embeddings is not None:
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(self.embeddings)
            
            model_info = model_data.get('model_info', {})
            print(f"✅ Enhanced model loaded successfully")
            print(f"📊 {len(self.qa_pairs)} Q&A pairs from {len(self.colleges_data)} colleges")
            print(f"📅 Created: {model_info.get('creation_date', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """Main training function"""
    print("🚀 Enhanced College AI Agent - English Training")
    print("=" * 60)
    
    # Initialize agent
    agent = EnhancedCollegeAIAgent()
    
    if len(agent.qa_pairs) == 0:
        print("❌ No training data found")
        return
    
    # Create embeddings
    agent.create_embeddings()
    
    # Save model
    agent.save_model()
    
    # Test the model
    print("\n🧪 Testing Enhanced Model:")
    print("-" * 40)
    
    test_queries = [
        "hi",
        "hello, how are you?",
        "what can you do?",
        "help me",
        "tell me about kalasalingam university",
        "what is the fee structure at IIT Bombay?",
        "which companies visit for placements?",
        "how to get admission in engineering colleges?",
        "what are the best engineering branches?",
        "compare government and private colleges",
        "thank you"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. 🔍 Query: '{query}'")
        results = agent.query_agent(query, top_k=1)
        
        if results:
            result = results[0]
            print(f"   ✅ Answer ({result['confidence']:.1f}%): {result['answer'][:150]}...")
        else:
            print("   ❌ No answer found")
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📊 Model contains {len(agent.qa_pairs)} Q&A pairs")
    print(f"🏫 Covering {len(agent.colleges_data)} engineering colleges")
    print("💾 Saved as 'enhanced_college_ai_english.pkl'")

if __name__ == "__main__":
    main()
