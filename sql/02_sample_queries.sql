-- ============================================================
-- KARE AI Chatbot - Useful Queries for MySQL Workbench
-- Run these after the app has some data
-- ============================================================

USE `kare_chatbot`;

-- ============================================================
-- USER QUERIES
-- ============================================================

-- List all registered users
SELECT id, name, email, created_at, last_login
FROM users
ORDER BY created_at DESC;

-- Count total users
SELECT COUNT(*) AS total_users FROM users;

-- Find a specific user by email
SELECT * FROM users WHERE email = 'example@gmail.com';

-- Users who logged in today
SELECT id, name, email, last_login
FROM users
WHERE DATE(last_login) = CURDATE();

-- Users with active sessions (currently logged in)
SELECT id, name, email, last_login
FROM users
WHERE session_token IS NOT NULL;


-- ============================================================
-- CHAT SESSION QUERIES
-- ============================================================

-- All chat sessions with user info
SELECT
    cs.id AS session_db_id,
    cs.session_id,
    cs.title,
    u.name AS user_name,
    u.email,
    cs.created_at,
    cs.updated_at
FROM chat_sessions cs
JOIN users u ON cs.user_id = u.id
ORDER BY cs.updated_at DESC;

-- Count sessions per user
SELECT
    u.name,
    u.email,
    COUNT(cs.id) AS total_sessions
FROM users u
LEFT JOIN chat_sessions cs ON u.id = cs.user_id
GROUP BY u.id
ORDER BY total_sessions DESC;

-- Recent sessions (last 24 hours)
SELECT
    cs.title,
    u.name,
    cs.created_at
FROM chat_sessions cs
JOIN users u ON cs.user_id = u.id
WHERE cs.created_at >= NOW() - INTERVAL 1 DAY
ORDER BY cs.created_at DESC;


-- ============================================================
-- CHAT MESSAGE QUERIES
-- ============================================================

-- All messages in a specific session (replace session_id value)
SELECT
    cm.role,
    cm.content,
    cm.language,
    cm.created_at
FROM chat_messages cm
WHERE cm.session_id = 1
ORDER BY cm.created_at ASC;

-- Full conversation view (session + messages)
SELECT
    cs.title AS session_title,
    u.name AS user_name,
    cm.role,
    LEFT(cm.content, 100) AS content_preview,
    cm.language,
    cm.created_at
FROM chat_messages cm
JOIN chat_sessions cs ON cm.session_id = cs.id
JOIN users u ON cm.user_id = u.id
ORDER BY cs.id, cm.created_at ASC;

-- Count messages per user
SELECT
    u.name,
    u.email,
    COUNT(cm.id) AS total_messages,
    SUM(CASE WHEN cm.role = 'user' THEN 1 ELSE 0 END) AS user_messages,
    SUM(CASE WHEN cm.role = 'bot' THEN 1 ELSE 0 END) AS bot_responses
FROM users u
LEFT JOIN chat_messages cm ON u.id = cm.user_id
GROUP BY u.id
ORDER BY total_messages DESC;

-- Most asked queries (top 20)
SELECT
    content AS question,
    language,
    COUNT(*) AS times_asked
FROM chat_messages
WHERE role = 'user'
GROUP BY content, language
ORDER BY times_asked DESC
LIMIT 20;

-- Messages by language
SELECT
    language,
    COUNT(*) AS message_count
FROM chat_messages
GROUP BY language
ORDER BY message_count DESC;

-- Today's chat activity
SELECT
    COUNT(DISTINCT cm.user_id) AS active_users,
    COUNT(DISTINCT cm.session_id) AS active_sessions,
    COUNT(*) AS total_messages
FROM chat_messages cm
WHERE DATE(cm.created_at) = CURDATE();


-- ============================================================
-- DASHBOARD / ANALYTICS QUERIES
-- ============================================================

-- Daily activity summary (last 7 days)
SELECT
    DATE(cm.created_at) AS chat_date,
    COUNT(DISTINCT cm.user_id) AS unique_users,
    COUNT(DISTINCT cm.session_id) AS sessions,
    COUNT(*) AS messages
FROM chat_messages cm
WHERE cm.created_at >= NOW() - INTERVAL 7 DAY
GROUP BY DATE(cm.created_at)
ORDER BY chat_date DESC;

-- Overall stats
SELECT
    (SELECT COUNT(*) FROM users) AS total_users,
    (SELECT COUNT(*) FROM chat_sessions) AS total_sessions,
    (SELECT COUNT(*) FROM chat_messages) AS total_messages,
    (SELECT COUNT(*) FROM chat_messages WHERE role = 'user') AS total_questions,
    (SELECT COUNT(*) FROM chat_messages WHERE role = 'bot') AS total_answers;


-- ============================================================
-- CLEANUP / MAINTENANCE QUERIES
-- ============================================================

-- Invalidate all session tokens (force re-login)
-- UPDATE users SET session_token = NULL;

-- Delete old sessions (older than 30 days)
-- DELETE FROM chat_sessions WHERE created_at < NOW() - INTERVAL 30 DAY;

-- Delete a specific user and all their data (cascades)
-- DELETE FROM users WHERE email = 'example@gmail.com';
