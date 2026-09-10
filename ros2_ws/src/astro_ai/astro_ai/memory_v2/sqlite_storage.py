"""ASTRO V1 — Thread-Safe SQLite3 Storage Engine for Memory V2.

Implements high-concurrency WAL mode, atomic transactions, and structured schemas
for Semantic Facts, Episodes, Relationships, Autobiographical events, and Spatial items.
"""

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class SQLiteMemoryStorage:
    """Thread-safe SQLite storage coordinator for Astro Cognitive Memory."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.abspath(os.path.join(here, "..", "..", "..", "..", "data"))
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, "astro_cognitive.db")
        elif db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = os.path.abspath(db_path)
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a thread-local SQLite connection configured with WAL mode."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            if self.db_path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        """Initializes database schema if not present."""
        conn = self._get_connection()
        with conn:
            # 1. Semantic Facts & Preferences
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    memory_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    value TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at REAL NOT NULL,
                    last_confirmed_at REAL NOT NULL,
                    last_used_at REAL DEFAULT 0,
                    use_count INTEGER DEFAULT 0,
                    evidence TEXT DEFAULT '',
                    visibility TEXT DEFAULT 'public',
                    importance REAL DEFAULT 0.5,
                    expires_at REAL,
                    contradiction_status TEXT DEFAULT 'active',
                    superseded_by_id TEXT,
                    created_by_person TEXT DEFAULT 'system'
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_subj_pred ON semantic_facts(subject, predicate);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_status ON semantic_facts(contradiction_status);")

            # 2. Relationship Profiles
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS relationship_profiles (
                    person_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    formal_title TEXT,
                    role TEXT NOT NULL,
                    familiarity REAL DEFAULT 0.0,
                    trust REAL DEFAULT 0.5,
                    interaction_count INTEGER DEFAULT 0,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    preferred_tone TEXT DEFAULT 'warm',
                    shared_topics_json TEXT DEFAULT '[]',
                    notes TEXT DEFAULT ''
                );
                """
            )

            # 3. Episodic Conversation Sessions
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_sessions (
                    session_id TEXT PRIMARY KEY,
                    person_id TEXT,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    turn_count INTEGER DEFAULT 0,
                    summary TEXT,
                    topics_json TEXT DEFAULT '[]',
                    emotional_arc TEXT
                );
                """
            )

            # 4. Autobiographical Memory (Robot's lived experiences)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS autobiographical_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    participants_json TEXT DEFAULT '[]',
                    location TEXT,
                    timestamp REAL NOT NULL,
                    emotional_valence REAL DEFAULT 0.0,
                    significance_score REAL DEFAULT 0.5
                );
                """
            )

            # 5. Spatial Memory (Landmarks & Objects)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spatial_landmarks (
                    item_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    relative_x_m REAL NOT NULL,
                    relative_y_m REAL NOT NULL,
                    orientation_deg REAL DEFAULT 0.0,
                    description TEXT DEFAULT '',
                    confidence REAL DEFAULT 1.0,
                    last_verified_ts REAL NOT NULL
                );
                """
            )

    def execute_write(self, query: str, params: Tuple = ()) -> int:
        """Executes an INSERT/UPDATE/DELETE query within an atomic transaction."""
        conn = self._get_connection()
        with conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount

    def execute_read(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Executes a SELECT query and returns rows."""
        conn = self._get_connection()
        cursor = conn.execute(query, params)
        return cursor.fetchall()
