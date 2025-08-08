from sentence_transformers import SentenceTransformer
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Load model once
_model = None

def get_model():
    """Lazily load the sentence transformer model"""
    global _model
    if _model is None:
        logger.info("Loading sentence transformer model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded successfully")
    return _model

def generate_embeddings(chunks):
    """Generate embeddings for list of chunk dictionaries (with 'text' key)"""
    if not chunks:
        return np.array([])

    try:
        model = get_model()
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(texts, show_progress_bar=False)

        return np.array(embeddings)

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return np.array([])

def embed_query(query: str) -> np.ndarray:
    """Embed a single user query"""
    try:
        model = get_model()
        return model.encode([query])[0]
    except Exception as e:
        logger.error(f"Error embedding query: {str(e)}")
        return np.zeros((384,))  # 384 is the dimension of MiniLM model
