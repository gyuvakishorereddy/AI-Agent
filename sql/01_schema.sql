-- ============================================================
-- KARE AI Chatbot - Database Schema
-- Run this in MySQL Workbench to create the full schema
-- ============================================================

-- Create database
CREATE DATABASE IF NOT EXISTS `kare_chatbot`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE `kare_chatbot`;

-- ============================================================
-- 1. Users Table
--    Stores registered users with hashed passwords
-- ============================================================
CREATE TABLE IF NOT EXISTS `users` (
    `id`             INT AUTO_INCREMENT PRIMARY KEY,
    `name`           VARCHAR(100) NOT NULL,
    `email`          VARCHAR(255) NOT NULL UNIQUE,
    `password_hash`  VARCHAR(128) NOT NULL,
    `password_salt`  VARCHAR(64)  NOT NULL,
    `session_token`  VARCHAR(128) DEFAULT NULL,
    `created_at`     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `last_login`     TIMESTAMP NULL,

    INDEX `idx_email`         (`email`),
    INDEX `idx_session_token` (`session_token`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 2. Chat Sessions Table
--    Groups messages into conversations per user
-- ============================================================
CREATE TABLE IF NOT EXISTS `chat_sessions` (
    `id`          INT AUTO_INCREMENT PRIMARY KEY,
    `user_id`     INT NOT NULL,
    `session_id`  VARCHAR(100) NOT NULL,
    `title`       VARCHAR(200) DEFAULT 'New Chat',
    `created_at`  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON DELETE CASCADE,
    INDEX `idx_user_id`    (`user_id`),
    INDEX `idx_session_id` (`session_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 3. Chat Messages Table
--    Stores every user query and bot response
-- ============================================================
CREATE TABLE IF NOT EXISTS `chat_messages` (
    `id`          INT AUTO_INCREMENT PRIMARY KEY,
    `session_id`  INT NOT NULL,
    `user_id`     INT NOT NULL,
    `role`        ENUM('user', 'bot') NOT NULL,
    `content`     TEXT NOT NULL,
    `language`    VARCHAR(10) DEFAULT 'en',
    `created_at`  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (`session_id`) REFERENCES `chat_sessions`(`id`) ON DELETE CASCADE,
    FOREIGN KEY (`user_id`)    REFERENCES `users`(`id`) ON DELETE CASCADE,
    INDEX `idx_session_id` (`session_id`),
    INDEX `idx_user_id`    (`user_id`),
    INDEX `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
