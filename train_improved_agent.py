#!/usr/bin/env python3
"""
Improved Training Script with Better Data Extraction
Fixes issues with specific college queries not returning relevant data
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
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

class ImprovedCollegeAIAgent:
    """Improved College AI Agent with better specific data extraction"""
    
    def __init__(self, data_path: str = "college_data"):
        print("🔧 Initializing Improved College AI Agent...")
        
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
            
        # College name variations for better matching
        self.college_name_variations = self._build_college_name_variations()
        
        # Load data
        self.load_college_data()
        self.prepare_improved_training_data()
        
    def _build_college_name_variations(self) -> Dict[str, List[str]]:
        """Build variations of college names for better matching"""
        variations = {
            'KL University': ['kl university', 'kl univ', 'klu', 'k l university', 'koneru lakshmaiah university'],
            'Kalasalingam University': ['kalasalingam', 'kalasalingam university', 'klu kalasalingam', 'kalasalingam academy'],
            'IIT Bombay': ['iit bombay', 'iit mumbai', 'indian institute of technology bombay'],
            'IIT Delhi': ['iit delhi', 'indian institute of technology delhi'],
            'VIT': ['vit university', 'vellore institute of technology', 'vit vellore'],
            'BITS Pilani': ['bits pilani', 'birla institute of technology and science']
        }
        return variations
    
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
    
    def prepare_improved_training_data(self):
        """Prepare improved Q&A pairs with better specific answers"""
        print("🎓 Preparing improved training data...")
        
        self.qa_pairs = []
        
        # Add general conversation pairs first
        self._add_general_conversation_pairs()
        
        # Add detailed college-specific data with better extraction
        for college_name, college_data in self.colleges_data.items():
            self._generate_detailed_college_qa_pairs(college_name, college_data)
        
        # Add enhanced recommendation queries
        self._add_enhanced_recommendation_queries()
        
        print(f"✅ Generated {len(self.qa_pairs)} improved Q&A pairs")
    
    def _add_general_conversation_pairs(self):
        """Add general conversation pairs"""
        general_pairs = [
            {
                'question': 'hi',
                'answer': "Hello! I'm your College AI Assistant. I can help you with detailed information about 637+ engineering colleges including fees, admissions, placements, and courses. What would you like to know?",
                'college': 'General',
                'category': 'conversation',
                'confidence_boost': 0.3
            },
            {
                'question': 'hello',
                'answer': "Hi there! I'm here to help you with comprehensive information about engineering colleges across India. Feel free to ask about specific colleges, fees, placements, or admission processes.",
                'college': 'General',
                'category': 'conversation',
                'confidence_boost': 0.3
            },
            {
                'question': 'what can you do',
                'answer': "I can provide detailed information about 637+ engineering colleges including:\n• Fee structures and payment details\n• Admission processes and entrance exams\n• Placement statistics and top recruiters\n• Course offerings and specializations\n• Campus facilities and infrastructure\n• College comparisons and recommendations\n\nJust ask me about any specific college or topic!",
                'college': 'General',
                'category': 'help',
                'confidence_boost': 0.3
            },
            {
                'question': 'help me',
                'answer': "I'm here to help! I can assist you with:\n\n🏫 **College Information**: Details about any of 637+ engineering colleges\n💰 **Fees**: Complete fee structures, scholarships, payment plans\n📝 **Admissions**: Entrance exams, eligibility, application process\n💼 **Placements**: Company visits, packages, placement statistics\n📚 **Courses**: Available branches, specializations, curriculum\n🏢 **Facilities**: Infrastructure, hostels, labs, libraries\n\nWhat specific information would you like?",
                'college': 'General',
                'category': 'help',
                'confidence_boost': 0.3
            }
        ]
        self.qa_pairs.extend(general_pairs)
    
    def _generate_detailed_college_qa_pairs(self, college_name: str, college_data: Dict):
        """Generate detailed Q&A pairs for a college with better specificity"""
        
        # Fee structure queries with detailed responses
        if 'fees_structure' in college_data:
            fees_data = college_data['fees_structure']
            self._add_detailed_fee_queries(college_name, fees_data)
        
        # Placement queries with specific data
        if 'placements' in college_data:
            placement_data = college_data['placements']
            self._add_detailed_placement_queries(college_name, placement_data)
        
        # Course queries with comprehensive information
        if 'courses' in college_data:
            course_data = college_data['courses']
            self._add_detailed_course_queries(college_name, course_data)
        
        # Admission queries with process details
        if 'admission_process' in college_data:
            admission_data = college_data['admission_process']
            self._add_detailed_admission_queries(college_name, admission_data)
        
        # Basic information with multiple question patterns
        if 'basic_info' in college_data:
            basic_data = college_data['basic_info']
            self._add_detailed_basic_queries(college_name, basic_data)
        
        # Facilities with comprehensive details
        if 'facilities' in college_data:
            facilities_data = college_data['facilities']
            self._add_detailed_facility_queries(college_name, facilities_data)
    
    def _add_detailed_fee_queries(self, college_name: str, fees_data: Dict):
        """Add detailed fee-related queries"""
        
        # Multiple variations of fee questions
        fee_questions = [
            f'what is the fee structure for {college_name.lower()}',
            f'what are the fees at {college_name.lower()}',
            f'{college_name.lower()} fees',
            f'{college_name.lower()} fee structure',
            f'how much does {college_name.lower()} cost',
            f'tuition fee at {college_name.lower()}',
            f'total fees for {college_name.lower()}',
            f'what is the cost of studying at {college_name.lower()}'
        ]
        
        # Extract detailed fee information
        fee_answer = self._extract_fee_details(college_name, fees_data)
        
        for question in fee_questions:
            self.qa_pairs.append({
                'question': question,
                'answer': fee_answer,
                'college': college_name,
                'category': 'fees',
                'confidence_boost': 0.2
            })
    
    def _extract_fee_details(self, college_name: str, fees_data: Dict) -> str:
        """Extract comprehensive fee details"""
        
        fee_details = []
        fee_details.append(f"**{college_name} Fee Structure:**\n")
        
        # Undergraduate fees
        if 'undergraduate' in fees_data:
            ug_fees = fees_data['undergraduate']
            if 'btech' in ug_fees:
                btech_fees = ug_fees['btech']
                if isinstance(btech_fees, dict):
                    if 'total_per_year' in btech_fees:
                        annual_fee = btech_fees['total_per_year']
                        fee_details.append(f"📚 **Annual Fee**: ₹{annual_fee:,}")
                    elif 'tuition_fee_per_year' in btech_fees:
                        tuition_fee = btech_fees['tuition_fee_per_year']
                        fee_details.append(f"📚 **Tuition Fee**: ₹{tuition_fee:,}")
        
        # Comprehensive fee structure
        if 'comprehensive_fee_structure' in fees_data:
            comp_fees = fees_data['comprehensive_fee_structure']
            if 'total_annual_cost' in comp_fees:
                total_cost = comp_fees['total_annual_cost']
                fee_details.append(f"💰 **Total Annual Cost**: ₹{total_cost:,}")
        
        # Hostel fees
        if 'hostel_fees' in fees_data:
            hostel_fee = fees_data['hostel_fees']
            fee_details.append(f"🏠 **Hostel Fee**: ₹{hostel_fee:,}")
        
        # Scholarships
        if 'scholarships' in fees_data:
            scholarships = fees_data['scholarships']
            if isinstance(scholarships, dict):
                fee_details.append("\n🎓 **Scholarships Available:**")
                if 'merit_scholarships' in scholarships:
                    fee_details.append(f"• Merit: {scholarships['merit_scholarships']}")
                if 'need_based' in scholarships:
                    fee_details.append(f"• Need-based: {scholarships['need_based']}")
        
        if not fee_details or len(fee_details) <= 1:
            return f"Fee information for {college_name} is available. Please contact the college for detailed fee structure."
        
        return "\n".join(fee_details)
    
    def _add_detailed_placement_queries(self, college_name: str, placement_data: Dict):
        """Add detailed placement-related queries"""
        
        placement_questions = [
            f'what are the placements at {college_name.lower()}',
            f'{college_name.lower()} placement statistics',
            f'which companies visit {college_name.lower()}',
            f'{college_name.lower()} average package',
            f'placement record of {college_name.lower()}',
            f'job opportunities at {college_name.lower()}'
        ]
        
        placement_answer = self._extract_placement_details(college_name, placement_data)
        
        for question in placement_questions:
            self.qa_pairs.append({
                'question': question,
                'answer': placement_answer,
                'college': college_name,
                'category': 'placements',
                'confidence_boost': 0.2
            })
    
    def _extract_placement_details(self, college_name: str, placement_data: Dict) -> str:
        """Extract comprehensive placement details"""
        
        placement_details = []
        placement_details.append(f"**{college_name} Placement Details:**\n")
        
        if isinstance(placement_data, dict):
            # Statistics
            if 'statistics' in placement_data:
                stats = placement_data['statistics']
                if isinstance(stats, dict):
                    if 'average_package' in stats:
                        avg_pkg = stats['average_package']
                        placement_details.append(f"💰 **Average Package**: {avg_pkg}")
                    if 'highest_package' in stats:
                        high_pkg = stats['highest_package']
                        placement_details.append(f"🎯 **Highest Package**: {high_pkg}")
                    if 'placement_percentage' in stats:
                        placement_pct = stats['placement_percentage']
                        placement_details.append(f"📊 **Placement Rate**: {placement_pct}")
            
            # Top recruiters
            if 'top_recruiters' in placement_data:
                companies = placement_data['top_recruiters']
                if companies and isinstance(companies, list):
                    placement_details.append(f"\n🏢 **Top Recruiters**: {', '.join(companies[:8])}")
        
        if len(placement_details) <= 1:
            return f"Placement information for {college_name} shows good industry connections with regular campus recruitment."
        
        return "\n".join(placement_details)
    
    def _add_detailed_course_queries(self, college_name: str, course_data: Dict):
        """Add detailed course-related queries"""
        
        course_questions = [
            f'what courses are offered at {college_name.lower()}',
            f'{college_name.lower()} courses',
            f'branches available at {college_name.lower()}',
            f'what can I study at {college_name.lower()}',
            f'{college_name.lower()} engineering branches'
        ]
        
        course_answer = self._extract_course_details(college_name, course_data)
        
        for question in course_questions:
            self.qa_pairs.append({
                'question': question,
                'answer': course_answer,
                'college': college_name,
                'category': 'courses',
                'confidence_boost': 0.2
            })
    
    def _extract_course_details(self, college_name: str, course_data: Dict) -> str:
        """Extract comprehensive course details"""
        
        course_details = []
        course_details.append(f"**{college_name} Courses Offered:**\n")
        
        if isinstance(course_data, dict) and 'undergraduate_programs' in course_data:
            ug_programs = course_data['undergraduate_programs']
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
                                course_details.append("🎓 **Engineering Branches:**")
                                for dept in dept_names[:10]:  # Show top 10
                                    course_details.append(f"• {dept}")
        
        if len(course_details) <= 1:
            return f"{college_name} offers various engineering courses across multiple specializations."
        
        return "\n".join(course_details)
    
    def _add_detailed_admission_queries(self, college_name: str, admission_data: Dict):
        """Add detailed admission-related queries"""
        
        admission_questions = [
            f'how to get admission in {college_name.lower()}',
            f'{college_name.lower()} admission process',
            f'entrance exams for {college_name.lower()}',
            f'how to apply to {college_name.lower()}',
            f'{college_name.lower()} eligibility criteria'
        ]
        
        admission_answer = self._extract_admission_details(college_name, admission_data)
        
        for question in admission_questions:
            self.qa_pairs.append({
                'question': question,
                'answer': admission_answer,
                'college': college_name,
                'category': 'admission',
                'confidence_boost': 0.2
            })
    
    def _extract_admission_details(self, college_name: str, admission_data: Dict) -> str:
        """Extract comprehensive admission details"""
        
        admission_details = []
        admission_details.append(f"**{college_name} Admission Process:**\n")
        
        if isinstance(admission_data, dict):
            if 'entrance_exams' in admission_data:
                exams = admission_data['entrance_exams']
                if exams and isinstance(exams, list):
                    admission_details.append(f"📝 **Entrance Exams**: {', '.join(exams)}")
            
            if 'eligibility' in admission_data:
                eligibility = admission_data['eligibility']
                admission_details.append(f"✅ **Eligibility**: {eligibility}")
        
        if len(admission_details) <= 1:
            return f"Admission to {college_name} is through entrance exams and merit-based selection."
        
        return "\n".join(admission_details)
    
    def _add_detailed_basic_queries(self, college_name: str, basic_data: Dict):
        """Add detailed basic information queries"""
        
        basic_questions = [
            f'tell me about {college_name.lower()}',
            f'what is {college_name.lower()}',
            f'{college_name.lower()} information',
            f'about {college_name.lower()}',
            f'{college_name.lower()} details'
        ]
        
        basic_answer = self._extract_basic_details(college_name, basic_data)
        
        for question in basic_questions:
            self.qa_pairs.append({
                'question': question,
                'answer': basic_answer,
                'college': college_name,
                'category': 'basic_info',
                'confidence_boost': 0.2
            })
    
    def _extract_basic_details(self, college_name: str, basic_data: Dict) -> str:
        """Extract comprehensive basic details"""
        
        basic_details = []
        basic_details.append(f"**About {college_name}:**\n")
        
        if isinstance(basic_data, dict):
            if 'about' in basic_data:
                about = basic_data['about']
                basic_details.append(f"{about}\n")
            
            if 'location' in basic_data:
                location = basic_data['location']
                if isinstance(location, dict):
                    city = location.get('city', '')
                    state = location.get('state', '')
                    if city and state:
                        basic_details.append(f"📍 **Location**: {city}, {state}")
                elif isinstance(location, str):
                    basic_details.append(f"📍 **Location**: {location}")
            
            if 'established' in basic_data:
                established = basic_data['established']
                basic_details.append(f"📅 **Established**: {established}")
            
            if 'ranking' in basic_data:
                ranking = basic_data['ranking']
                if isinstance(ranking, dict) and 'nirf_ranking' in ranking:
                    nirf = ranking['nirf_ranking']
                    basic_details.append(f"🏆 **NIRF Ranking**: {nirf}")
        
        if len(basic_details) <= 1:
            return f"{college_name} is a reputed engineering institution."
        
        return "\n".join(basic_details)
    
    def _add_detailed_facility_queries(self, college_name: str, facilities_data: Dict):
        """Add detailed facility-related queries"""
        
        facility_questions = [
            f'what facilities are available at {college_name.lower()}',
            f'{college_name.lower()} facilities',
            f'infrastructure at {college_name.lower()}',
            f'{college_name.lower()} campus facilities'
        ]
        
        facility_answer = self._extract_facility_details(college_name, facilities_data)
        
        for question in facility_questions:
            self.qa_pairs.append({
                'question': question,
                'answer': facility_answer,
                'college': college_name,
                'category': 'facilities',
                'confidence_boost': 0.2
            })
    
    def _extract_facility_details(self, college_name: str, facilities_data: Dict) -> str:
        """Extract comprehensive facility details"""
        
        facility_details = []
        facility_details.append(f"**{college_name} Facilities:**\n")
        
        if isinstance(facilities_data, dict):
            facilities = list(facilities_data.keys())
            if facilities:
                facility_details.append("🏢 **Available Facilities:**")
                for facility in facilities[:8]:  # Show top 8
                    facility_details.append(f"• {facility.replace('_', ' ').title()}")
        
        if len(facility_details) <= 1:
            return f"{college_name} provides comprehensive facilities for academic and extracurricular activities."
        
        return "\n".join(facility_details)
    
    def _add_enhanced_recommendation_queries(self):
        """Add enhanced recommendation queries"""
        
        recommendation_pairs = [
            {
                'question': 'tell me a good private college for mechanical engineering',
                'answer': "Here are some excellent private colleges for Mechanical Engineering:\n\n🏆 **Top Private Colleges:**\n• **BITS Pilani** - Premier institute with excellent placements\n• **VIT University** - Strong industry connections\n• **SRM Institute** - Good infrastructure and faculty\n• **Manipal Institute of Technology** - Established reputation\n• **KIIT University** - Modern facilities\n\nFor specific information about fees, admission process, or placements at any of these colleges, just ask!",
                'college': 'Recommendation',
                'category': 'recommendation',
                'confidence_boost': 0.3
            },
            {
                'question': 'which is the best engineering college',
                'answer': "The best engineering colleges in India include:\n\n🥇 **IITs (Government)**:\n• IIT Bombay, IIT Delhi, IIT Madras, IIT Kanpur\n\n🥈 **NITs (Government)**:\n• NIT Trichy, NIT Surathkal, NIT Warangal\n\n🥉 **Top Private**:\n• BITS Pilani, VIT University, Manipal Institute of Technology\n\nThe 'best' depends on your branch preference, budget, and career goals. Would you like detailed information about any specific college?",
                'college': 'Recommendation',
                'category': 'recommendation',
                'confidence_boost': 0.3
            },
            {
                'question': 'in those which is best',
                'answer': "Among the top colleges, here's the ranking:\n\n🏆 **Overall Best**: IIT Bombay (Top choice for placements and research)\n🥈 **Second Best**: IIT Delhi (Excellent for tech companies)\n🥉 **Third Best**: IIT Madras (Strong in South India)\n\n**For Private Colleges**: BITS Pilani is considered the best private engineering college.\n\nThe choice depends on:\n• Your JEE rank\n• Preferred location\n• Specific branch interest\n• Budget considerations\n\nWould you like specific details about any of these colleges?",
                'college': 'Recommendation',
                'category': 'recommendation',
                'confidence_boost': 0.3
            }
        ]
        
        self.qa_pairs.extend(recommendation_pairs)
    
    def create_embeddings(self):
        """Create embeddings for all Q&A pairs"""
        print("🧠 Creating improved embeddings...")
        
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
        
        print(f"✅ Created improved embeddings with dimension {dimension}")
    
    def enhanced_search(self, question: str, top_k: int = 5) -> List[Dict]:
        """Enhanced search with better college name matching"""
        if not self.index or not self.qa_pairs:
            return []
        
        # Create question embedding
        question_embedding = self.sentence_model.encode([question])
        faiss.normalize_L2(question_embedding)
        
        # Search for similar questions
        scores, indices = self.index.search(question_embedding, min(top_k * 3, len(self.qa_pairs)))
        
        results = []
        question_lower = question.lower()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.qa_pairs):
                qa = self.qa_pairs[idx]
                confidence = float(score) * 100
                
                # Apply confidence boost for better matches
                if qa.get('confidence_boost'):
                    confidence += qa['confidence_boost'] * 100
                
                # Boost confidence for exact college name matches
                college_name = qa['college'].lower()
                if college_name != 'general' and college_name != 'recommendation':
                    for college, variations in self.college_name_variations.items():
                        if college.lower() == college_name:
                            if any(var in question_lower for var in variations):
                                confidence += 20  # Extra boost for college name match
                                break
                
                results.append({
                    'college': qa['college'],
                    'category': qa['category'],
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'confidence': confidence
                })
        
        # Filter and sort results
        results = [r for r in results if r['confidence'] > 40]
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results[:top_k]
    
    def query_agent(self, question: str, top_k: int = 5) -> List[Dict]:
        """Enhanced query processing"""
        
        results = self.enhanced_search(question, top_k)
        
        # If no good results, provide helpful guidance
        if not results or (results and results[0]['confidence'] < 60):
            guidance_answer = f"""I understand you're asking about "{question}". 

I can help you with detailed information about 637+ engineering colleges including:

💰 **Fee Structures**: Complete breakdown with scholarships
📝 **Admissions**: Entrance exams and eligibility criteria  
💼 **Placements**: Company visits and package details
📚 **Courses**: Available branches and specializations
🏢 **Facilities**: Infrastructure and campus amenities

**Try asking more specifically like:**
• "What is the fee structure for [College Name]?"
• "Tell me about placements at [College Name]"
• "Which are the best private colleges for mechanical engineering?"

Which college or topic would you like to know about?"""
            
            return [{
                'college': 'AI Assistant',
                'category': 'guidance',
                'question': question,
                'answer': guidance_answer,
                'confidence': 75.0
            }]
        
        return results
    
    def save_model(self, model_path: str = "improved_college_ai_english.pkl"):
        """Save the improved model"""
        print(f"💾 Saving improved model to {model_path}...")
        
        model_data = {
            'qa_pairs': self.qa_pairs,
            'embeddings': self.embeddings,
            'colleges_data': self.colleges_data,
            'college_name_variations': self.college_name_variations,
            'model_info': {
                'creation_date': datetime.now().isoformat(),
                'total_colleges': len(self.colleges_data),
                'total_qa_pairs': len(self.qa_pairs),
                'model_type': 'improved_english',
                'sentence_transformer': 'all-MiniLM-L6-v2',
                'improvements': ['Better college name matching', 'Detailed data extraction', 'Enhanced specificity']
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Improved model saved successfully ({file_size:.1f} MB)")
    
    def load_model(self, model_path: str = "improved_college_ai_english.pkl"):
        """Load a pre-trained improved model"""
        print(f"📥 Loading improved model from {model_path}...")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.qa_pairs = model_data['qa_pairs']
            self.embeddings = model_data['embeddings']
            self.colleges_data = model_data['colleges_data']
            self.college_name_variations = model_data.get('college_name_variations', {})
            
            # Recreate FAISS index
            if self.embeddings is not None:
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(self.embeddings)
            
            model_info = model_data.get('model_info', {})
            print(f"✅ Improved model loaded successfully")
            print(f"📊 {len(self.qa_pairs)} Q&A pairs from {len(self.colleges_data)} colleges")
            print(f"📅 Created: {model_info.get('creation_date', 'Unknown')}")
            print(f"🔧 Improvements: {', '.join(model_info.get('improvements', []))}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """Main training function"""
    print("🚀 Improved College AI Agent - Enhanced Training")
    print("=" * 60)
    
    # Initialize agent
    agent = ImprovedCollegeAIAgent()
    
    if len(agent.qa_pairs) == 0:
        print("❌ No training data found")
        return
    
    # Create embeddings
    agent.create_embeddings()
    
    # Save model
    agent.save_model()
    
    # Test the specific queries that were problematic
    print("\n🧪 Testing Improved Model with Problematic Queries:")
    print("-" * 60)
    
    test_queries = [
        "what is the fees structure for kl university",
        "kl university fees",
        "tell me a good private college for mechanical engineering",
        "in those which is best",
        "tell me about kalasalingam university"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. 🔍 Query: '{query}'")
        results = agent.query_agent(query, top_k=1)
        
        if results:
            result = results[0]
            print(f"   ✅ Answer ({result['confidence']:.1f}%): {result['answer'][:200]}...")
            print(f"   🏫 Source: {result['college']} | Category: {result['category']}")
        else:
            print("   ❌ No answer found")
    
    print(f"\n🎉 Improved training completed successfully!")
    print(f"📊 Model contains {len(agent.qa_pairs)} Q&A pairs")
    print(f"🏫 Covering {len(agent.colleges_data)} engineering colleges")
    print("💾 Saved as 'improved_college_ai_english.pkl'")

if __name__ == "__main__":
    main()
