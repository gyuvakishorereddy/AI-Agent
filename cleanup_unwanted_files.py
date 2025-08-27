#!/usr/bin/env python3
"""
AI Agent Project Cleanup - Remove Unwanted Files
Clean up redundant, obsolete, and temporary files while keeping essential components
"""

import os
import shutil
from pathlib import Path

def cleanup_unwanted_files():
    """Remove unwanted files and keep only essential components"""
    
    base_path = Path("c:/Users/kishore/Music/ai agent")
    
    # Files to remove - categorized by type
    files_to_remove = {
        # Development/experimental files that are no longer needed
        "development_files": [
            "add_ap_private_colleges.py",
            "add_real_engineering_colleges.py", 
            "ai_agent_data_enhancer.py",
            "batch_college_data_generator.py",
            "college_data_generator.py",
            "complete_data_generator.py",
            "comprehensive_500_colleges_database.py",
            "comprehensive_college_data_generator.py",
            "expand_to_300_colleges.py",
            "expand_to_500_colleges.py",
            "top_300_colleges_generator.py",
            "top_500_colleges_generator.py",
        ],
        
        # Fix/audit scripts that have completed their purpose
        "completed_fix_scripts": [
            "complete_qa_audit_and_fix.py",
            "complete_real_names_fix.py",
            "comprehensive_answer_fix.py",
            "comprehensive_data_analysis.py",
            "comprehensive_data_audit.py",
            "comprehensive_data_fix.py",
            "comprehensive_recheck.py",
            "comprehensive_wrong_answer_fix.py",
            "database_cleanup_and_update.py",
            "data_validation_and_correction.py",
            "final_14_names_fix.py",
            "final_cleanup_all_generic_names.py",
            "final_comprehensive_answer_fix.py",
            "final_comprehensive_fix.py",
            "final_real_names_replacement.py",
            "final_specific_answer_fix.py",
            "final_verification.py",
            "fix_all_faq_answers.py",
            "fix_all_wrong_answers.py",
            "fix_college_data_structure.py",
            "fix_question_answer_matching.py",
            "fix_real_college_names.py",
            "fix_remaining_issues.py",
            "perfect_question_answer_matching.py",
        ],
        
        # Old model files and backups
        "old_models": [
            "college_ai_agent_fixed.pkl",
            "college_ai_agent_old.pkl",
            "train_college_ai_agent_backup.py",
        ],
        
        # Redundant training scripts
        "redundant_training": [
            "train_gpu_optimized.py",
            "train_rtx2050_gpu.py", 
            "train_with_language_models.py",
            "load_and_run_h5.py",
            "verify_cuda.py",
        ],
        
        # Integration and demo files
        "demo_integration": [
            "demo_dynamic_updates.py",
            "integrate_json_data.py",
            "json_data_loader.py",
            "multilingual_demo.py",
            "offline_dynamic_ai.py",
            "simple_test.py",
            "test_multilingual.py",
            "voice_chat_app.py",
        ],
        
        # Utility scripts that have served their purpose
        "utility_scripts": [
            "generate_missing_models.py",
            "improved_query_handler.py", 
            "intelligent_faq_generator.py",
            "update_query_logic.py",
            "verify_college_data.py",
        ],
        
        # Old update scripts (keeping only latest)
        "old_update_scripts": [
            "comprehensive_2025_data_audit.py",
            "update_latest_data_2025.py",
        ],
        
        # Temporary test files
        "test_files": [
            "test_improved_agent.py",
        ],
        
        # Report files that are now consolidated
        "old_reports": [
            "AI_AGENT_ENHANCEMENT_REPORT.md",
            "AP_COLLEGES_ADDITION_REPORT.md",
            "COMPLETE_AUDIT_REPORT.md",
            "COMPREHENSIVE_ANSWER_FIX_REPORT.md",
            "COMPREHENSIVE_RECHECK_REPORT.md",
            "DATABASE_CLEANUP_REPORT.md",
            "database_completion_summary.md",
            "DATABASE_MAINTENANCE_SUMMARY.md",
            "FAQ_IMPROVEMENT_REPORT.md",
            "FINAL_COMPLETION_REPORT.md",
            "FINAL_COMPREHENSIVE_AUDIT_REPORT.md",
            "FINAL_DATABASE_COMPLETION_REPORT.md",
            "FINAL_FAQ_FIX_REPORT.md",
            "SYSTEM_FIXES_REPORT.md",
            "college_database_summary.md",
        ],
        
        # Redundant text files
        "text_files": [
            "all_colleges_data.txt",
            "training_data.json",
        ],
        
        # Old notebooks (keeping only essential ones)
        "old_notebooks": [
            "colab code.ipynb",
            "kalasalingam_chatbot_enhanced.ipynb",
            "multi_college_chatbot.ipynb",
        ]
    }
    
    # Essential files to KEEP
    essential_files = {
        "core_functionality": [
            "train_college_ai_agent.py",  # Main training script
            "test_trained_agent.py",      # Essential testing
            "college_ai_agent.pkl",       # Primary model
            "college_ai_multilingual_ready.pkl",  # Multilingual model
            "api_server_multilingual.py", # API server
            "multi_college_manager.py",   # Core manager
            "dynamic_college_ai.py",      # Dynamic functionality
            "system_health_check.py",     # Health monitoring
        ],
        "data_and_config": [
            "requirements.txt",
            "install_requirements.py",
            "install_multilingual_requirements.py",
        ],
        "documentation": [
            "README.md",
            "README_MULTILINGUAL.md", 
            "MULTILINGUAL_README.md",
            "AI_AGENT_TRAINING_GUIDE.md",
            "COMPREHENSIVE_ANALYSIS_AND_FIXES_REPORT.md",
            "COMPREHENSIVE_2025_AUDIT_REPORT.md",
            "COMPREHENSIVE_2025_DATA_UPDATE_REPORT.md",
        ],
        "notebooks": [
            "College_AI_Agent_Training.ipynb",  # Main training notebook
        ],
        "current_scripts": [
            "comprehensive_2025_data_updater.py",  # Latest update script
        ],
        "directories": [
            "college_data/",
            "multilingual_data/",
            "templates/",
            "college_ai_deployment/",
            "dynamic_cache/",
        ]
    }
    
    print("🧹 AI Agent Project Cleanup")
    print("=" * 50)
    
    # Count files to be removed
    total_files = 0
    for category in files_to_remove.values():
        total_files += len(category)
    
    print(f"📊 Analysis: {total_files} files identified for removal")
    
    # Remove files by category
    removed_count = 0
    
    for category_name, file_list in files_to_remove.items():
        print(f"\n🗂️  Removing {category_name}:")
        category_removed = 0
        
        for filename in file_list:
            file_path = base_path / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"   ✅ Removed: {filename}")
                    removed_count += 1
                    category_removed += 1
                except Exception as e:
                    print(f"   ❌ Error removing {filename}: {e}")
            else:
                print(f"   ⚠️  Not found: {filename}")
        
        print(f"   📊 Category total: {category_removed} files removed")
    
    # Clean up __pycache__ directories
    print(f"\n🧹 Cleaning __pycache__ directories:")
    pycache_paths = list(base_path.rglob("__pycache__"))
    for pycache_path in pycache_paths:
        try:
            shutil.rmtree(pycache_path)
            print(f"   ✅ Removed: {pycache_path}")
            removed_count += 1
        except Exception as e:
            print(f"   ❌ Error removing {pycache_path}: {e}")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"   🗑️  Total files removed: {removed_count}")
    print(f"   ✅ Essential files preserved")
    print(f"   🎯 Project cleaned and optimized")
    
    # Show remaining essential files
    print(f"\n📋 Essential Files Preserved:")
    for category_name, file_list in essential_files.items():
        if category_name != "directories":
            print(f"\n   {category_name}:")
            for filename in file_list:
                file_path = base_path / filename
                if file_path.exists():
                    size = file_path.stat().st_size
                    size_mb = size / (1024*1024)
                    print(f"     ✅ {filename} ({size_mb:.1f} MB)")
                else:
                    print(f"     ⚠️  {filename} (missing)")
    
    return removed_count

if __name__ == "__main__":
    try:
        removed = cleanup_unwanted_files()
        print(f"\n🎉 Cleanup completed successfully!")
        print(f"   💾 Storage space freed up")
        print(f"   🎯 Project structure optimized")
        print(f"   ✅ Ready for production use")
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        import traceback
        traceback.print_exc()
