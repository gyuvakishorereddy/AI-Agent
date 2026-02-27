"""
MySQL Database Module for KARE AI Chatbot
Handles: user authentication, chat history storage
"""

import mysql.connector
from mysql.connector import Error
import hashlib
import secrets
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

# ----- Config from .env -----
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
}
DB_NAME = os.getenv("DB_NAME", "kare_chatbot")
JWT_SECRET = os.getenv("JWT_SECRET", "default_secret_key")


# ----- Helpers -----
def _hash_password(password: str, salt: str = None) -> tuple:
    """Hash password with SHA-256 + salt. Returns (hash, salt)."""
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return hashed, salt


def _generate_token(user_id: int, email: str) -> str:
    """Generate a simple session token."""
    raw = f"{user_id}:{email}:{secrets.token_hex(16)}:{JWT_SECRET}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _validate_gmail(email: str) -> bool:
    """Validate that email is a @gmail.com address."""
    pattern = r'^[a-zA-Z0-9._%+-]+@gmail\.com$'
    return bool(re.match(pattern, email, re.IGNORECASE))


# ----- Connection pool -----
_pool = None


def _get_connection():
    """Get a MySQL connection."""
    global _pool
    try:
        if _pool is None:
            _pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="kare_pool",
                pool_size=5,
                pool_reset_session=True,
                host=DB_CONFIG["host"],
                port=DB_CONFIG["port"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                database=DB_NAME,
                autocommit=True,
            )
        return _pool.get_connection()
    except Error as e:
        logger.error(f"DB connection error: {e}")
        raise


# ----- Schema initialization -----
def init_database():
    """Create database and tables if they don't exist."""
    conn = None
    try:
        # First connect without database to create it
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
        )
        cursor = conn.cursor()

        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.execute(f"USE `{DB_NAME}`")

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                password_hash VARCHAR(128) NOT NULL,
                password_salt VARCHAR(64) NOT NULL,
                session_token VARCHAR(128) DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL,
                INDEX idx_email (email),
                INDEX idx_session_token (session_token)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)

        # Chat sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                session_id VARCHAR(100) NOT NULL,
                title VARCHAR(200) DEFAULT 'New Chat',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_user_id (user_id),
                INDEX idx_session_id (session_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)

        # Chat messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id INT NOT NULL,
                user_id INT NOT NULL,
                role ENUM('user', 'bot') NOT NULL,
                content TEXT NOT NULL,
                language VARCHAR(10) DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_session_id (session_id),
                INDEX idx_user_id (user_id),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)

        conn.commit()
        logger.info("✅ Database and tables initialized successfully")
        return True

    except Error as e:
        logger.error(f"❌ Database initialization error: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


# ----- User operations -----
def signup_user(name: str, email: str, password: str) -> dict:
    """Register a new user. Email must be @gmail.com."""
    if not name or not name.strip():
        return {"success": False, "message": "Name is required"}

    if not _validate_gmail(email):
        return {"success": False, "message": "Email must be a valid @gmail.com address"}

    if len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters"}

    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)

        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (email.lower(),))
        if cursor.fetchone():
            return {"success": False, "message": "An account with this email already exists"}

        # Hash password and insert
        pw_hash, pw_salt = _hash_password(password)
        token = _generate_token(0, email)

        cursor.execute(
            "INSERT INTO users (name, email, password_hash, password_salt, session_token, last_login) "
            "VALUES (%s, %s, %s, %s, %s, NOW())",
            (name.strip(), email.lower(), pw_hash, pw_salt, token)
        )
        conn.commit()

        user_id = cursor.lastrowid
        # Update token with real user_id
        token = _generate_token(user_id, email)
        cursor.execute("UPDATE users SET session_token = %s WHERE id = %s", (token, user_id))
        conn.commit()

        logger.info(f"✅ User registered: {email}")
        return {
            "success": True,
            "message": "Account created successfully!",
            "user": {
                "id": user_id,
                "name": name.strip(),
                "email": email.lower(),
                "token": token,
            }
        }

    except Error as e:
        logger.error(f"Signup error: {e}")
        return {"success": False, "message": "Registration failed. Please try again."}
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def signin_user(email: str, password: str) -> dict:
    """Authenticate a user."""
    if not _validate_gmail(email):
        return {"success": False, "message": "Email must be a valid @gmail.com address"}

    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT id, name, email, password_hash, password_salt FROM users WHERE email = %s",
            (email.lower(),)
        )
        user = cursor.fetchone()

        if not user:
            return {"success": False, "message": "No account found with this email"}

        # Verify password
        pw_hash, _ = _hash_password(password, user["password_salt"])
        if pw_hash != user["password_hash"]:
            return {"success": False, "message": "Incorrect password"}

        # Generate new session token
        token = _generate_token(user["id"], email)
        cursor.execute(
            "UPDATE users SET session_token = %s, last_login = NOW() WHERE id = %s",
            (token, user["id"])
        )
        conn.commit()

        logger.info(f"✅ User signed in: {email}")
        return {
            "success": True,
            "message": "Signed in successfully!",
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "token": token,
            }
        }

    except Error as e:
        logger.error(f"Signin error: {e}")
        return {"success": False, "message": "Sign in failed. Please try again."}
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def validate_token(token: str) -> dict:
    """Validate a session token. Returns user info or None."""
    if not token:
        return None
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, name, email FROM users WHERE session_token = %s",
            (token,)
        )
        user = cursor.fetchone()
        return user
    except Error as e:
        logger.error(f"Token validation error: {e}")
        return None
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def logout_user(token: str) -> bool:
    """Invalidate session token."""
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET session_token = NULL WHERE session_token = %s", (token,))
        conn.commit()
        return True
    except Error as e:
        logger.error(f"Logout error: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


# ----- Chat operations -----
def create_chat_session(user_id: int, session_id: str, title: str = "New Chat") -> int:
    """Create a new chat session. Returns session DB id."""
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_sessions (user_id, session_id, title) VALUES (%s, %s, %s)",
            (user_id, session_id, title)
        )
        conn.commit()
        return cursor.lastrowid
    except Error as e:
        logger.error(f"Create session error: {e}")
        return None
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def get_or_create_session(user_id: int, session_id: str) -> int:
    """Get existing session DB id or create new one."""
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id FROM chat_sessions WHERE user_id = %s AND session_id = %s",
            (user_id, session_id)
        )
        row = cursor.fetchone()
        if row:
            return row["id"]
        # Create new
        cursor.execute(
            "INSERT INTO chat_sessions (user_id, session_id) VALUES (%s, %s)",
            (user_id, session_id)
        )
        conn.commit()
        return cursor.lastrowid
    except Error as e:
        logger.error(f"Get/create session error: {e}")
        return None
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def save_message(user_id: int, session_db_id: int, role: str, content: str, language: str = "en"):
    """Save a chat message to the database."""
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_messages (session_id, user_id, role, content, language) VALUES (%s, %s, %s, %s, %s)",
            (session_db_id, user_id, role, content, language)
        )
        conn.commit()

        # Update session title from first user message
        cursor.execute(
            "SELECT title FROM chat_sessions WHERE id = %s", (session_db_id,)
        )
        row = cursor.fetchone()
        if row and row[0] == "New Chat" and role == "user":
            title = content[:50] + ("..." if len(content) > 50 else "")
            cursor.execute(
                "UPDATE chat_sessions SET title = %s WHERE id = %s",
                (title, session_db_id)
            )
            conn.commit()

        return True
    except Error as e:
        logger.error(f"Save message error: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def get_user_chat_sessions(user_id: int, limit: int = 50) -> list:
    """Get all chat sessions for a user, ordered by most recent."""
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, session_id, title, created_at, updated_at "
            "FROM chat_sessions WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s",
            (user_id, limit)
        )
        sessions = cursor.fetchall()
        # Convert datetime objects to strings
        for s in sessions:
            s["created_at"] = s["created_at"].isoformat() if s["created_at"] else None
            s["updated_at"] = s["updated_at"].isoformat() if s["updated_at"] else None
        return sessions
    except Error as e:
        logger.error(f"Get sessions error: {e}")
        return []
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def get_session_messages(session_db_id: int, user_id: int) -> list:
    """Get all messages for a chat session."""
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT role, content, language, created_at "
            "FROM chat_messages WHERE session_id = %s AND user_id = %s ORDER BY created_at ASC",
            (session_db_id, user_id)
        )
        messages = cursor.fetchall()
        for m in messages:
            m["created_at"] = m["created_at"].isoformat() if m["created_at"] else None
        return messages
    except Error as e:
        logger.error(f"Get messages error: {e}")
        return []
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def delete_chat_session(session_db_id: int, user_id: int) -> bool:
    """Delete a chat session and its messages."""
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM chat_sessions WHERE id = %s AND user_id = %s",
            (session_db_id, user_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    except Error as e:
        logger.error(f"Delete session error: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


# ----- Init on import -----
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Initializing KARE Chatbot Database...")
    if init_database():
        print("✅ Database ready!")
    else:
        print("❌ Database initialization failed!")
