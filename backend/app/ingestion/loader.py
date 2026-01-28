"""
Document Loader - Load documents from various formats.

Supports PDF, Markdown, and plain text files with YAML frontmatter.
"""

import logging
import os
import re
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Document loader supporting multiple formats.
    
    Handles PDF, Markdown, and text files with optional YAML frontmatter.
    """
    
    SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".markdown"}
    
    def __init__(self):
        """Initialize the document loader."""
        pass
    
    def load_file(self, filepath: str) -> tuple[str, dict]:
        """
        Load a document from a file.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            Tuple of (content, frontmatter_dict)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == ".pdf":
            return self._load_pdf(filepath), {}
        elif ext in {".md", ".markdown", ".txt"}:
            return self._load_text_with_frontmatter(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def load_from_bytes(
        self, 
        content: bytes, 
        filename: Optional[str] = None
    ) -> str:
        """
        Load document content from bytes.
        
        Args:
            content: File content as bytes
            filename: Optional filename to determine format
            
        Returns:
            Extracted text content
        """
        # Determine type from filename or content
        if filename:
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".pdf":
                return self._load_pdf_bytes(content)
        
        # Default: treat as text
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1")
    
    def _load_text_with_frontmatter(self, filepath: str) -> tuple[str, dict]:
        """
        Load a text file with optional YAML frontmatter.
        
        Frontmatter is delimited by --- at the start of the file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        return self._parse_frontmatter(content)
    
    def _parse_frontmatter(self, content: str) -> tuple[str, dict]:
        """
        Parse YAML frontmatter from content.
        
        Args:
            content: Document content with potential frontmatter
            
        Returns:
            Tuple of (content_without_frontmatter, frontmatter_dict)
        """
        frontmatter = {}
        
        # Check for frontmatter delimiter
        if content.startswith("---"):
            # Find the closing delimiter
            end_match = re.search(r"\n---\s*\n", content[3:])
            if end_match:
                frontmatter_text = content[3:end_match.start() + 3]
                try:
                    frontmatter = yaml.safe_load(frontmatter_text) or {}
                except yaml.YAMLError as e:
                    logger.warning(f"Failed to parse frontmatter: {e}")
                
                # Remove frontmatter from content
                content = content[end_match.end() + 3:].strip()
        
        return content, frontmatter
    
    def _load_pdf(self, filepath: str) -> str:
        """
        Load content from a PDF file.
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            from PyPDF2 import PdfReader
            
            reader = PdfReader(filepath)
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n\n".join(text_parts)
            
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")
    
    def _load_pdf_bytes(self, content: bytes) -> str:
        """
        Load content from PDF bytes.
        
        Args:
            content: PDF file content as bytes
            
        Returns:
            Extracted text content
        """
        try:
            from PyPDF2 import PdfReader
            import io
            
            reader = PdfReader(io.BytesIO(content))
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n\n".join(text_parts)
            
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")
    
    def list_documents(self, directory: str) -> list[str]:
        """
        List all supported documents in a directory.
        
        Args:
            directory: Path to directory
            
        Returns:
            List of file paths
        """
        documents = []
        
        for root, dirs, files in os.walk(directory):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    documents.append(os.path.join(root, filename))
        
        return sorted(documents)
