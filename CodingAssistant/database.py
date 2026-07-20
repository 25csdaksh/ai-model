import sqlite3
import os
import json
from datetime import datetime

# Database file path
DB_FILE = os.path.join(os.path.dirname(__file__), "coding_assistant.db")

def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables for chat sessions, messages, and code snippets."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Conversations Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE NOT NULL,
        title TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Chat Messages Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES conversations (session_id) ON DELETE CASCADE
    );
    """)

    # 3. Saved Code Snippets Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS saved_snippets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        language TEXT NOT NULL,
        code TEXT NOT NULL,
        tags TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully at:", DB_FILE)

# --- Helper Functions ---

def save_message(session_id: str, role: str, content: str, title: str = "New Chat Session"):
    """Save a chat message to the database, creating a session if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Ensure conversation session exists
    cursor.execute("SELECT id FROM conversations WHERE session_id = ?", (session_id,))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO conversations (session_id, title) VALUES (?, ?)", (session_id, title))

    # Insert message
    cursor.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()

def get_chat_history(session_id: str):
    """Retrieve all messages for a specific chat session."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def save_code_snippet(title: str, language: str, code: str, tags: str = ""):
    """Save a code snippet to the repository."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO saved_snippets (title, language, code, tags) VALUES (?, ?, ?, ?)",
                   (title, language, code, tags))
    conn.commit()
    snippet_id = cursor.lastrowid
    conn.close()
    return snippet_id

def get_all_snippets():
    """Get all saved code snippets."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM saved_snippets ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

if __name__ == "__main__":
    init_db()
