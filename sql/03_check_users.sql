-- ============================================================
-- KARE AI - Check Users and Debug Database
-- Run this in MySQL Workbench to verify user data
-- ============================================================

-- 1. Select the correct database
USE kare_chatbot;

-- 2. Show all tables
SHOW TABLES;

-- ============================================================
-- 3. CHECK USERS TABLE
-- ============================================================

-- Count total users
SELECT COUNT(*) as 'Total Users' FROM users;

-- Show all users with details
SELECT 
    id,
    name,
    email,
    password_hash,
    session_token,
    created_at,
    last_login
FROM users
ORDER BY created_at DESC;

-- Show users in simple format
SELECT 
    id,
    name,
    email,
    created_at
FROM users;

-- ============================================================
-- 4. CHECK CHAT SESSIONS TABLE
-- ============================================================

-- Count total sessions
SELECT COUNT(*) as 'Total Sessions' FROM chat_sessions;

-- Show all sessions with user info
SELECT 
    cs.id,
    cs.user_id,
    u.email,
    cs.title,
    cs.created_at,
    cs.updated_at
FROM chat_sessions cs
LEFT JOIN users u ON cs.user_id = u.id
ORDER BY cs.created_at DESC;

-- ============================================================
-- 5. CHECK CHAT MESSAGES TABLE
-- ============================================================

-- Count total messages
SELECT COUNT(*) as 'Total Messages' FROM chat_messages;

-- Show recent messages
SELECT 
    cm.id,
    cm.user_id,
    u.email,
    cm.role,
    cm.content,
    cm.language,
    cm.created_at
FROM chat_messages cm
LEFT JOIN users u ON cm.user_id = u.id
ORDER BY cm.created_at DESC
LIMIT 20;

-- ============================================================
-- 6. FIND A SPECIFIC USER
-- ============================================================

-- Replace 'user@gmail.com' with your actual email
SELECT 
    id,
    name,
    email,
    created_at,
    last_login
FROM users
WHERE email = 'user@gmail.com';

-- ============================================================
-- 7. CHECK USER'S CHAT HISTORY
-- ============================================================

-- Replace user_id with actual ID (e.g., 1)
SELECT 
    cs.id as session_id,
    cs.title,
    cs.created_at,
    COUNT(cm.id) as message_count
FROM chat_sessions cs
LEFT JOIN chat_messages cm ON cs.id = cm.session_id
WHERE cs.user_id = 1
GROUP BY cs.id, cs.title, cs.created_at;

-- ============================================================
-- 8. VIEW USER'S MESSAGES
-- ============================================================

-- Replace user_id with actual ID (e.g., 1)
SELECT 
    cm.id,
    cm.role,
    SUBSTRING(cm.content, 1, 100) as message_preview,
    cm.language,
    cm.created_at
FROM chat_messages cm
WHERE cm.user_id = 1
ORDER BY cm.created_at DESC
LIMIT 50;

-- ============================================================
-- 9. DATABASE STRUCTURE
-- ============================================================

-- Show users table structure
DESCRIBE users;

-- Show chat_sessions table structure
DESCRIBE chat_sessions;

-- Show chat_messages table structure
DESCRIBE chat_messages;

-- ============================================================
-- 10. VERIFY FOREIGN KEY RELATIONSHIPS
-- ============================================================

-- Check if any session has invalid user_id
SELECT cs.id, cs.user_id
FROM chat_sessions cs
WHERE cs.user_id NOT IN (SELECT id FROM users);

-- Check if any message has invalid user_id
SELECT cm.id, cm.user_id
FROM chat_messages cm
WHERE cm.user_id NOT IN (SELECT id FROM users);

-- ============================================================
-- USEFUL COMMANDS
-- ============================================================

-- Delete all data (be careful!)
-- DELETE FROM chat_messages;
-- DELETE FROM chat_sessions;
-- DELETE FROM users;

-- Reset auto-increment
-- ALTER TABLE users AUTO_INCREMENT = 1;
-- ALTER TABLE chat_sessions AUTO_INCREMENT = 1;
-- ALTER TABLE chat_messages AUTO_INCREMENT = 1;

-- ============================================================
