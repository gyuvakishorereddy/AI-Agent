#!/usr/bin/env python
"""
KARE AI Chatbot - Complete Startup Script
Handles: JSON→MD conversion, Vector store building, Server startup
"""

import os
import sys
import json
import pickle
from pathlib import Path
import subprocess
from datetime import datetime


class ChatbotBootstrapper:
    """Complete initialization for KARE AI Chatbot"""
    
    def __init__(self):
        self.root_dir = Path.cwd()
        self.data_dir = self.root_dir / "data"
        self.data_md_dir = self.root_dir / "data_md"
        
    def log(self, message: str, level: str = "INFO"):
        """Colored logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        colors = {
            "INFO": "\033[94m",      # Blue
            "SUCCESS": "\033[92m",   # Green
            "WARNING": "\033[93m",   # Yellow
            "ERROR": "\033[91m",     # Red
        }
        reset = "\033[0m"
        
        color = colors.get(level, colors["INFO"])
        print(f"{color}[{timestamp}] {level:8} | {message}{reset}")
    
    def check_dependencies(self):
        """Check if all required packages are installed"""
        self.log("Checking dependencies...", "INFO")
        
        required = {
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "scikit-learn": "sklearn",
            "sentence-transformers": "sentence_transformers",
            "pydantic": "pydantic"
        }
        
        missing = []
        for package, import_name in required.items():
            try:
                __import__(import_name)
                self.log(f"✓ {package}", "SUCCESS")
            except ImportError:
                self.log(f"✗ {package} missing", "WARNING")
                missing.append(package)
        
        if missing:
            self.log(f"Installing missing packages: {', '.join(missing)}", "WARNING")
            os.system(f"pip install {' '.join(missing)} --quiet")
        
        return True
    
    def convert_json_to_markdown(self):
        """Convert JSON files to markdown"""
        self.log("Step 1: Converting JSON to Markdown", "INFO")
        
        if self.data_md_dir.exists():
            md_files = list(self.data_md_dir.glob("*.md"))
            self.log(f"Found {len(md_files)} existing markdown files", "SUCCESS")
            return True
        
        self.log("Converting JSON files...", "INFO")
        
        try:
            # Import conversion script
            sys.path.insert(0, str(self.root_dir))
            from convert_json_to_md import json_to_markdown
            
            self.data_md_dir.mkdir(exist_ok=True)
            
            json_files = list(self.data_dir.glob("*.json"))
            for json_file in sorted(json_files):
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                title = json_file.stem.replace('_', ' ').title()
                md_content = json_to_markdown(json_data, title)
                
                md_file = self.data_md_dir / f"{json_file.stem}.md"
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                
                self.log(f"✓ {json_file.name}", "SUCCESS")
            
            self.log(f"Converted {len(json_files)} files to markdown", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Error converting JSON: {str(e)}", "ERROR")
            return False
    
    def build_vector_store(self):
        """Build or load vector store"""
        self.log("Step 2: Building Vector Store", "INFO")
        
        vector_store_dir = self.root_dir / "faiss_index"
        
        if vector_store_dir.exists():
            required_files = [
                vector_store_dir / "faiss_index.bin",
                vector_store_dir / "metadata.pkl"
            ]
            
            if all(f.exists() for f in required_files):
                self.log("Vector store already exists", "SUCCESS")
                
                # Show stats
                try:
                    with open(vector_store_dir / "metadata.pkl", 'rb') as f:
                        metadata = pickle.load(f)
                        chunks = metadata.get('chunks', [])
                        self.log(f"Loaded {len(chunks)} chunks from existing store", "SUCCESS")
                except:
                    self.log("Vector store exists (stats unavailable)", "SUCCESS")
                return True
        
        self.log("Building new vector store...", "INFO")
        
        try:
            sys.path.insert(0, str(self.root_dir / "src"))
            from vector_store import VectorStoreManager
            
            # Initialize vector store manager
            self.log("Initializing vector store manager...", "INFO")
            manager = VectorStoreManager(
                data_dir=str(self.data_md_dir),
                vector_store_path=str(vector_store_dir)
            )
            
            # Build vector store
            self.log("Building FAISS vector store (this may take a few minutes)...", "INFO")
            success = manager.build_vector_store(force_rebuild=True)
            
            if success:
                self.log(f"Vector store built successfully with {len(manager.chunks)} chunks", "SUCCESS")
                return True
            else:
                self.log("Failed to build vector store", "ERROR")
                return False
            
        except Exception as e:
            self.log(f"Error building vector store: {str(e)}", "ERROR")
            return False
    
    def verify_setup(self):
        """Verify all components are ready"""
        self.log("Step 3: Verifying Setup", "INFO")
        
        vector_store_dir = self.root_dir / "faiss_index"
        
        checks = {
            "Markdown files": self.data_md_dir.exists() and len(list(self.data_md_dir.glob("*.md"))) > 0,
            "Vector store": all([
                (vector_store_dir / "faiss_index.bin").exists(),
                (vector_store_dir / "metadata.pkl").exists(),
            ]),
            "App file": (self.root_dir / "src" / "app.py").exists(),
            "Public assets": (self.root_dir / "public").exists(),
        }
        
        all_ready = True
        for check, result in checks.items():
            status = "✓" if result else "✗"
            level = "SUCCESS" if result else "ERROR"
            self.log(f"{status} {check}", level)
            if not result:
                all_ready = False
        
        return all_ready
    
    def start_server(self):
        """Start FastAPI server"""
        self.log("Step 4: Starting Server", "INFO")
        
        try:
            self.log("Launching FastAPI server on http://localhost:8000", "INFO")
            self.log("=" * 70, "INFO")
            
            # Change to src directory and run app
            app_path = self.root_dir / "src" / "app.py"
            if not app_path.exists():
                # Try app_simple.py as fallback
                app_path = self.root_dir / "src" / "app_simple.py"
            
            # Properly quote the path for Windows
            os.system(f'python "{app_path}"')
            
        except KeyboardInterrupt:
            self.log("Server stopped by user", "WARNING")
        except Exception as e:
            self.log(f"Server error: {str(e)}", "ERROR")
    
    def run(self):
        """Complete startup sequence"""
        print("\n" + "=" * 70)
        print("🚀 KARE AI Chatbot Bootstrap")
        print("=" * 70 + "\n")
        
        try:
            # Step 1: Check dependencies
            if not self.check_dependencies():
                self.log("Dependency check failed", "ERROR")
                return False
            
            print()
            
            # Step 2: Convert JSON to Markdown
            if not self.convert_json_to_markdown():
                self.log("JSON to Markdown conversion failed", "ERROR")
                return False
            
            print()
            
            # Step 3: Build vector store
            if not self.build_vector_store():
                self.log("Vector store building failed", "ERROR")
                return False
            
            print()
            
            # Step 4: Verify setup
            if not self.verify_setup():
                self.log("Setup verification failed", "ERROR")
                return False
            
            print("\n" + "=" * 70)
            self.log("✨ All systems ready! Starting server...", "SUCCESS")
            print("=" * 70 + "\n")
            
            # Step 5: Start server
            self.start_server()
            
        except Exception as e:
            self.log(f"Bootstrap failed: {str(e)}", "ERROR")
            return False
        
        return True


if __name__ == "__main__":
    bootstrapper = ChatbotBootstrapper()
    success = bootstrapper.run()
    sys.exit(0 if success else 1)
