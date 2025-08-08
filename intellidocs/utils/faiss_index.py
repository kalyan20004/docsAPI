import faiss
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def create_faiss_index(embeddings: np.ndarray):
    """
    Create a FAISS index from normalized embeddings using cosine similarity.

    Args:
        embeddings: A 2D NumPy array of shape (n_samples, dim)

    Returns:
        FAISS index object
    """
    if embeddings.size == 0:
        logger.error("No embeddings provided to create_faiss_index")
        return None

    try:
        # Ensure float32 and normalize for cosine similarity
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product ≈ cosine similarity for normalized vectors

        index.add(embeddings)
        logger.info(f"FAISS index created with {index.ntotal} vectors (dim={dim})")

        return index

    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        return None

def retrieve_top_k_chunks(query_embedding: np.ndarray, index, chunks: List[Dict], k: int = 5) -> List[Dict]:
    """
    Retrieve top-k similar chunks based on a query embedding.

    Args:
        query_embedding: NumPy array of shape (dim,) for the user query
        index: FAISS index
        chunks: Original list of text chunks (with metadata)
        k: Number of top results to return

    Returns:
        List of top-k chunk dicts with similarity scores added
    """
    if index is None:
        logger.error("FAISS index is None")
        return []
        
    if len(chunks) == 0:
        logger.error("No chunks provided")
        return []

    try:
        # Ensure query embedding is 2D and normalized
        if query_embedding.ndim == 1:
            query_vector = query_embedding.reshape(1, -1)
        else:
            query_vector = query_embedding
            
        query_vector = query_vector.astype("float32")
        faiss.normalize_L2(query_vector)

        # Search for top k chunks
        num_search = min(k, len(chunks), index.ntotal)
        scores, indices = index.search(query_vector, num_search)

        top_chunks = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx >= 0 and idx < len(chunks):  # Valid index
                chunk = chunks[idx].copy()
                chunk['similarity_score'] = float(score)
                chunk['rank'] = rank + 1
                top_chunks.append(chunk)

        logger.info(f"Retrieved {len(top_chunks)} chunks with scores: {[c['similarity_score'] for c in top_chunks[:3]]}")
        return top_chunks

    except Exception as e:
        logger.error(f"Error retrieving top chunks: {str(e)}")
        return []

# For backward compatibility, keep the old function name
def query_faiss_index(index, query, chunks, k=5):
    """Backward compatibility wrapper"""
    from utils.embedder import embed_query
    
    try:
        query_embedding = embed_query(query)
        return retrieve_top_k_chunks(query_embedding, index, chunks, k)
    except Exception as e:
        logger.error(f"Error in query_faiss_index: {str(e)}")
        return []