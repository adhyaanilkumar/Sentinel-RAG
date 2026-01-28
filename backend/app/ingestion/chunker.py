"""
Semantic Chunker - Split documents into meaningful chunks.

Uses section-based chunking optimized for military document formats.
"""

import logging
import re
from typing import Any

from app.models.documents import DocumentChunk

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Semantic chunker that splits documents at natural boundaries.
    
    Optimized for military document formats with clear section structure.
    """
    
    # Section header patterns for military documents
    SECTION_PATTERNS = [
        r'^#{1,3}\s+',          # Markdown headers
        r'^[A-Z][A-Z\s]+:$',    # ALL CAPS headers with colon
        r'^[A-Z][A-Z\s]+$',     # ALL CAPS headers
        r'^\d+\.\s+[A-Z]',      # Numbered sections
        r'^[IVX]+\.\s+',        # Roman numeral sections
    ]
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Compile section patterns
        self._section_regex = re.compile(
            '|'.join(f'({p})' for p in self.SECTION_PATTERNS),
            re.MULTILINE
        )
    
    def chunk(
        self, 
        document: str, 
        metadata: dict[str, Any]
    ) -> list[DocumentChunk]:
        """
        Split a document into semantic chunks.
        
        Attempts section-based chunking first, falls back to
        size-based chunking if sections are too large.
        
        Args:
            document: Document text content
            metadata: Base metadata for all chunks
            
        Returns:
            List of document chunks
        """
        if not document.strip():
            return []
        
        # Try section-based chunking
        sections = self._split_by_sections(document)
        
        # Process sections into appropriately-sized chunks
        chunks = []
        for i, section in enumerate(sections):
            section_chunks = self._process_section(section, metadata, i, len(sections))
            chunks.extend(section_chunks)
        
        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _split_by_sections(self, document: str) -> list[str]:
        """
        Split document at section headers.
        
        Returns list of sections including their headers.
        """
        sections = []
        current_section = []
        
        for line in document.split('\n'):
            # Check if this is a section header
            if self._is_section_header(line):
                # Save current section if non-empty
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """Check if a line is a section header."""
        line = line.strip()
        if not line:
            return False
        return bool(self._section_regex.match(line))
    
    def _process_section(
        self, 
        section: str, 
        base_metadata: dict, 
        section_index: int,
        total_sections: int,
    ) -> list[DocumentChunk]:
        """
        Process a section into one or more chunks.
        
        Large sections are split further by size.
        """
        section = section.strip()
        if not section:
            return []
        
        # Extract section header if present
        lines = section.split('\n')
        section_header = ""
        if lines and self._is_section_header(lines[0]):
            section_header = lines[0].strip()
        
        # If section is small enough, return as single chunk
        if len(section) <= self.chunk_size:
            return [DocumentChunk(
                content=section,
                metadata={
                    **base_metadata,
                    "chunk_index": section_index,
                    "total_chunks": total_sections,
                    "section_header": section_header,
                }
            )]
        
        # Split large section by size with overlap
        return self._split_by_size(section, base_metadata, section_header)
    
    def _split_by_size(
        self, 
        text: str, 
        base_metadata: dict,
        section_header: str = "",
    ) -> list[DocumentChunk]:
        """
        Split text into chunks by size with overlap.
        
        Tries to split at sentence boundaries.
        """
        chunks = []
        
        # Split into sentences for cleaner breaks
        sentences = self._split_sentences(text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds limit, save current chunk
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata={
                        **base_metadata,
                        "chunk_index": len(chunks),
                        "section_header": section_header,
                    }
                ))
                
                # Keep overlap sentences
                overlap_text = chunk_text[-self.chunk_overlap:] if self.chunk_overlap else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
            
            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space
        
        # Add remaining content
        if current_chunk:
            chunks.append(DocumentChunk(
                content=' '.join(current_chunk),
                metadata={
                    **base_metadata,
                    "chunk_index": len(chunks),
                    "section_header": section_header,
                }
            ))
        
        # Update total_chunks in metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.
        
        Uses common sentence terminators and handles abbreviations.
        """
        # Simple sentence splitting - handles common cases
        # More sophisticated NLP could be used here
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _merge_small_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """
        Merge chunks that are too small.
        
        Small chunks at the end of sections are merged with previous.
        """
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # If current chunk is small and not the last
            if len(current.content) < self.min_chunk_size and i < len(chunks) - 1:
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged_content = current.content + "\n\n" + next_chunk.content
                merged.append(DocumentChunk(
                    content=merged_content,
                    metadata={**next_chunk.metadata, "merged": True}
                ))
                i += 2
            elif len(current.content) < self.min_chunk_size and merged:
                # Merge with previous chunk
                prev = merged.pop()
                merged_content = prev.content + "\n\n" + current.content
                merged.append(DocumentChunk(
                    content=merged_content,
                    metadata={**prev.metadata, "merged": True}
                ))
                i += 1
            else:
                merged.append(current)
                i += 1
        
        return merged
