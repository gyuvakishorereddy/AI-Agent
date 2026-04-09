"""
Quick script to check if users are being saved to the database
Run: python check_users.py
"""

import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "kare_chatbot"),
}

try:
    print("=" * 70)
    print("KARE AI - DATABASE USER CHECKER")
    print("=" * 70)
    
    # Connect to MySQL
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    
    # Check database
    print(f"\n✅ Connected to database: {DB_CONFIG['database']}")
    
    # List tables
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    print(f"\n📋 Tables in database:")
    for table in tables:
        table_name = list(table.values())[0]
        print(f"   - {table_name}")
    
    # Get user count
    cursor.execute("SELECT COUNT(*) as count FROM users")
    result = cursor.fetchone()
    user_count = result['count']
    print(f"\n👥 Total users in database: {user_count}")
    
    # Show all users
    if user_count > 0:
        print(f"\n📝 All users:")
        print("-" * 70)
        cursor.execute("SELECT id, name, email, created_at, last_login FROM users ORDER BY created_at DESC")
        users = cursor.fetchall()
        for user in users:
            print(f"  ID: {user['id']}")
            print(f"  Name: {user['name']}")
            print(f"  Email: {user['email']}")
            print(f"  Created: {user['created_at']}")
            print(f"  Last Login: {user['last_login']}")
            print("-" * 70)
    else:
        print("\n⚠️  No users found in database yet!")
        print("   Try signing up a new user in the app.")
    
    # Check chat_sessions
    cursor.execute("SELECT COUNT(*) as count FROM chat_sessions")
    session_count = cursor.fetchone()['count']
    print(f"\n💬 Total chat sessions: {session_count}")
    
    cursor.close()
    conn.close()
    
    print("\n✅ Check complete!")
    print("=" * 70)

except Error as e:
    print(f"\n❌ Database connection error: {e}")
    print("   Make sure MySQL is running and .env credentials are correct")
except Exception as e:
    print(f"❌ Error: {e}")
