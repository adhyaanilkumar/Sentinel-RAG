"""
Knowledge Base Versioning - Track KB versions for reproducibility.

Computes deterministic hashes of knowledge base content.
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def compute_kb_hash(kb_path: str) -> str:
    """
    Compute a hash of the entire knowledge base for versioning.
    
    Ensures reproducibility by tracking exact KB state.
    
    Args:
        kb_path: Path to knowledge base directory
        
    Returns:
        12-character hex hash of KB contents
    """
    if not os.path.exists(kb_path):
        return "empty"
    
    hasher = hashlib.sha256()
    
    # Sort for deterministic ordering
    for root, dirs, files in os.walk(kb_path):
        dirs.sort()  # Ensure consistent directory order
        for filename in sorted(files):
            filepath = os.path.join(root, filename)
            
            # Include relative path in hash for structure awareness
            rel_path = os.path.relpath(filepath, kb_path)
            hasher.update(rel_path.encode())
            
            # Include file contents
            try:
                with open(filepath, 'rb') as f:
                    hasher.update(f.read())
            except (IOError, OSError) as e:
                logger.warning(f"Could not read file for hashing: {filepath}: {e}")
    
    return hasher.hexdigest()[:12]


def get_kb_version(kb_path: str) -> dict:
    """
    Get comprehensive version information for knowledge base.
    
    Args:
        kb_path: Path to knowledge base directory
        
    Returns:
        Dictionary with version info
    """
    if not os.path.exists(kb_path):
        return {
            "hash": "empty",
            "document_count": 0,
            "last_modified": None,
            "categories": {},
        }
    
    # Compute hash
    kb_hash = compute_kb_hash(kb_path)
    
    # Count documents and categories
    document_count = 0
    categories = {}
    last_modified = None
    
    for root, dirs, files in os.walk(kb_path):
        rel_path = os.path.relpath(root, kb_path)
        category = rel_path.split(os.sep)[0] if rel_path != "." else "root"
        
        for filename in files:
            if filename.endswith(('.md', '.txt')):
                document_count += 1
                categories[category] = categories.get(category, 0) + 1
                
                # Track most recent modification
                filepath = os.path.join(root, filename)
                mtime = os.path.getmtime(filepath)
                if last_modified is None or mtime > last_modified:
                    last_modified = mtime
    
    return {
        "hash": kb_hash,
        "document_count": document_count,
        "last_modified": datetime.fromtimestamp(last_modified).isoformat() if last_modified else None,
        "categories": categories,
    }


def save_version_manifest(kb_path: str, output_path: Optional[str] = None) -> str:
    """
    Save a version manifest file for the knowledge base.
    
    Args:
        kb_path: Path to knowledge base
        output_path: Path for manifest file (default: kb_path/VERSION.json)
        
    Returns:
        Path to saved manifest
    """
    version_info = get_kb_version(kb_path)
    version_info["generated_at"] = datetime.now().isoformat()
    
    if output_path is None:
        output_path = os.path.join(kb_path, "VERSION.json")
    
    with open(output_path, 'w') as f:
        json.dump(version_info, f, indent=2)
    
    logger.info(f"Saved KB version manifest: {output_path}")
    return output_path


def verify_kb_version(kb_path: str, expected_hash: str) -> bool:
    """
    Verify that KB matches expected version hash.
    
    Args:
        kb_path: Path to knowledge base
        expected_hash: Expected hash value
        
    Returns:
        True if hashes match
    """
    current_hash = compute_kb_hash(kb_path)
    matches = current_hash == expected_hash
    
    if not matches:
        logger.warning(
            f"KB version mismatch: expected {expected_hash}, got {current_hash}"
        )
    
    return matches
