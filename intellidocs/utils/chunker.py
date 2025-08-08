import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def chunk_text(text: str, source_file: str = "", chunk_size: int = 300, overlap: int = 50, min_chunk_size: int = 50) -> List[Dict]:
    """
    Split text into overlapping chunks with metadata
    
    Args:
        text: Input text to chunk
        source_file: Name of source file
        chunk_size: Maximum words per chunk
        overlap: Number of overlapping words between chunks
        min_chunk_size: Minimum words required for a chunk
        
    Returns:
        List of dictionaries with text and metadata
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking")
        return []
    
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text.strip())
    words = text.split()
    
    if len(words) <= chunk_size:
        if len(words) >= min_chunk_size:
            return [{
                'text': text,
                'source': source_file,
                'chunk_id': 0,
                'start_word': 0,
                'end_word': len(words),
                'word_count': len(words)
            }]
        else:
            return []
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        
        if len(chunk_words) >= min_chunk_size:
            chunk_text = " ".join(chunk_words)
            chunks.append({
                'text': chunk_text,
                'source': source_file,
                'chunk_id': chunk_id,
                'start_word': start,
                'end_word': end,
                'word_count': len(chunk_words)
            })
            chunk_id += 1
        
        if end >= len(words):
            break
        
        start += chunk_size - overlap
    
    logger.info(f"Created {len(chunks)} chunks from {len(words)} words (source: {source_file})")
    return chunks