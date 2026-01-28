"""
Response caching for demo reliability.

Caches API responses to ensure demos work even if OpenAI is unavailable.
"""

import hashlib
import json
import sqlite3
import os
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    SQLite-based response cache for LLM and vision API calls.
    
    Ensures demo reliability by caching responses for reuse.
    """
    
    def __init__(self, db_path: str = "data/response_cache.db"):
        """Initialize the cache with SQLite database."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                operation_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1
            )
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_cache_operation 
            ON cache(operation_type)
        ''')
        self.conn.commit()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Create a deterministic string representation
        key_data = json.dumps({
            "args": [str(a)[:1000] for a in args],  # Truncate large args
            "kwargs": {k: str(v)[:1000] for k, v in sorted(kwargs.items())}
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        """
        Get a cached response by key.
        
        Returns None if not found.
        """
        cursor = self.conn.execute(
            "SELECT response FROM cache WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        
        if row:
            # Update access tracking
            self.conn.execute(
                """UPDATE cache 
                   SET accessed_at = ?, access_count = access_count + 1 
                   WHERE key = ?""",
                (datetime.now().isoformat(), key)
            )
            self.conn.commit()
            logger.debug(f"Cache hit for key: {key[:16]}...")
            return row[0]
        
        logger.debug(f"Cache miss for key: {key[:16]}...")
        return None
    
    def set(self, key: str, response: str, operation_type: str = "unknown"):
        """
        Store a response in the cache.
        
        Overwrites existing entries with the same key.
        """
        self.conn.execute(
            """INSERT OR REPLACE INTO cache 
               (key, response, operation_type, created_at, accessed_at, access_count)
               VALUES (?, ?, ?, ?, ?, 1)""",
            (key, response, operation_type, 
             datetime.now().isoformat(), datetime.now().isoformat())
        )
        self.conn.commit()
        logger.debug(f"Cached response for key: {key[:16]}...")
    
    def get_or_set(
        self, 
        key: str, 
        generator_func, 
        operation_type: str = "unknown"
    ) -> str:
        """
        Get from cache or generate and cache.
        
        If not in cache, calls generator_func to create the response.
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        
        # Generate new response
        response = generator_func()
        self.set(key, response, operation_type)
        return response
    
    def clear(self):
        """Clear all cached responses."""
        self.conn.execute("DELETE FROM cache")
        self.conn.commit()
        logger.info("Cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_entries,
                COUNT(DISTINCT operation_type) as operation_types,
                SUM(access_count) as total_accesses
            FROM cache
        """)
        row = cursor.fetchone()
        
        # Get counts by operation type
        cursor = self.conn.execute("""
            SELECT operation_type, COUNT(*) as count
            FROM cache
            GROUP BY operation_type
        """)
        by_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        return {
            "total_entries": row[0],
            "operation_types": row[1],
            "total_accesses": row[2],
            "by_operation_type": by_type,
        }
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
