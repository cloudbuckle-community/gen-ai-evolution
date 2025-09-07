import numpy as np
from typing import List
import hashlib


class EmbeddingsModel:
    """Simple embeddings model for demonstration purposes"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for text using a simple hash-based approach"""
        # In production, use actual embedding models like sentence-transformers
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Convert hash to numeric values
        numeric_values = [ord(c) for c in text_hash]

        # Pad or truncate to desired dimension
        if len(numeric_values) < self.dimension:
            numeric_values.extend([0] * (self.dimension - len(numeric_values)))
        else:
            numeric_values = numeric_values[:self.dimension]

        # Normalize to unit vector
        embedding = np.array(numeric_values, dtype=np.float32)
        return embedding / np.linalg.norm(embedding)

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple documents"""
        return [self.embed_text(text) for text in texts]