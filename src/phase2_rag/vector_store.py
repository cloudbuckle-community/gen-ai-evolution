# src/phase2_rag/vector_store.py
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Document class for storing content with embeddings and metadata"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate document after initialization"""
        if not self.id:
            raise ValueError("Document ID cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")
        if self.metadata is None:
            self.metadata = {}


class VectorStore:
    """
    In-memory vector store for RAG implementation
    Supports adding documents, similarity search, and vector operations
    """

    def __init__(self, similarity_threshold: float = 0.0):
        """
        Initialize vector store

        Args:
            similarity_threshold: Minimum similarity score for search results
        """
        self.documents: Dict[str, Document] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.document_ids: List[str] = []
        self.similarity_threshold = similarity_threshold

    def add_document(self, doc: Document) -> None:
        """
        Add a single document to the vector store

        Args:
            doc: Document to add

        Raises:
            ValueError: If document has no embedding
        """
        if doc.embedding is None:
            raise ValueError(f"Document {doc.id} must have an embedding")

        self.documents[doc.id] = doc
        self._rebuild_embeddings_matrix()

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add multiple documents to the vector store

        Args:
            docs: List of documents to add
        """
        for doc in docs:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} must have an embedding")
            self.documents[doc.id] = doc

        self._rebuild_embeddings_matrix()

    def similarity_search(self,
                          query_embedding: np.ndarray,
                          k: int = 3,
                          filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Find the k most similar documents to the query

        Args:
            query_embedding: Query vector to search for
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of most similar documents
        """
        if self.embeddings_matrix is None or len(self.documents) == 0:
            return []

        if query_embedding.shape[0] != self.embeddings_matrix.shape[1]:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[0]} "
                f"doesn't match stored embeddings dimension {self.embeddings_matrix.shape[1]}"
            )

        # Compute similarities
        similarities = self._compute_similarities(query_embedding)

        # Apply similarity threshold
        valid_indices = np.where(similarities >= self.similarity_threshold)[0]

        if len(valid_indices) == 0:
            return []

        # Sort by similarity (highest first)
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]

        # Apply metadata filtering if specified
        results = []
        for idx in sorted_indices:
            if idx < len(self.document_ids):
                doc = self.documents[self.document_ids[idx]]

                # Check metadata filters
                if filter_metadata and not self._matches_metadata_filter(doc, filter_metadata):
                    continue

                results.append(doc)

                # Stop when we have enough results
                if len(results) >= k:
                    break

        return results

    def similarity_search_with_scores(self,
                                      query_embedding: np.ndarray,
                                      k: int = 3) -> List[Tuple[Document, float]]:
        """
        Find similar documents and return with similarity scores

        Args:
            query_embedding: Query vector to search for
            k: Number of results to return

        Returns:
            List of (document, similarity_score) tuples
        """
        if self.embeddings_matrix is None or len(self.documents) == 0:
            return []

        similarities = self._compute_similarities(query_embedding)
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            if idx < len(self.document_ids) and similarities[idx] >= self.similarity_threshold:
                doc = self.documents[self.document_ids[idx]]
                score = float(similarities[idx])
                results.append((doc, score))

        return results

    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID

        Args:
            doc_id: Document ID to retrieve

        Returns:
            Document if found, None otherwise
        """
        return self.documents.get(doc_id)

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the vector store

        Args:
            doc_id: Document ID to remove

        Returns:
            True if document was removed, False if not found
        """
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._rebuild_embeddings_matrix()
            return True
        return False

    def update_document(self, doc: Document) -> bool:
        """
        Update an existing document

        Args:
            doc: Updated document

        Returns:
            True if document was updated, False if not found
        """
        if doc.id in self.documents:
            self.documents[doc.id] = doc
            self._rebuild_embeddings_matrix()
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics

        Returns:
            Dictionary with store statistics
        """
        if not self.documents:
            return {
                "total_documents": 0,
                "embedding_dimension": 0,
                "memory_usage_mb": 0
            }

        # Calculate memory usage (approximate)
        memory_usage = 0
        if self.embeddings_matrix is not None:
            memory_usage = self.embeddings_matrix.nbytes / (1024 * 1024)  # MB

        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embeddings_matrix.shape[1] if self.embeddings_matrix is not None else 0,
            "memory_usage_mb": round(memory_usage, 2),
            "similarity_threshold": self.similarity_threshold
        }

    def clear(self) -> None:
        """Clear all documents from the vector store"""
        self.documents.clear()
        self.document_ids.clear()
        self.embeddings_matrix = None

    def _rebuild_embeddings_matrix(self) -> None:
        """Rebuild the embeddings matrix when documents are added/removed"""
        if not self.documents:
            self.embeddings_matrix = None
            self.document_ids = []
            return

        self.document_ids = list(self.documents.keys())
        embeddings = []

        for doc_id in self.document_ids:
            embedding = self.documents[doc_id].embedding
            if embedding is not None:
                embeddings.append(embedding)
            else:
                raise ValueError(f"Document {doc_id} has no embedding")

        if embeddings:
            self.embeddings_matrix = np.array(embeddings)
        else:
            self.embeddings_matrix = None

    def _compute_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between query and all documents

        Args:
            query_embedding: Query vector

        Returns:
            Array of similarity scores
        """
        if self.embeddings_matrix is None:
            return np.array([])

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(self.document_ids))

        normalized_query = query_embedding / query_norm

        # Normalize document embeddings
        doc_norms = np.linalg.norm(self.embeddings_matrix, axis=1)

        # Handle zero vectors
        non_zero_mask = doc_norms != 0
        similarities = np.zeros(len(self.document_ids))

        if np.any(non_zero_mask):
            normalized_docs = self.embeddings_matrix[non_zero_mask] / doc_norms[non_zero_mask][:, np.newaxis]
            similarities[non_zero_mask] = np.dot(normalized_docs, normalized_query)

        return similarities

    def _matches_metadata_filter(self, doc: Document, filter_metadata: Dict[str, Any]) -> bool:
        """
        Check if document matches metadata filters

        Args:
            doc: Document to check
            filter_metadata: Metadata filters to apply

        Returns:
            True if document matches all filters
        """
        for key, value in filter_metadata.items():
            if key not in doc.metadata:
                return False

            doc_value = doc.metadata[key]

            # Handle different comparison types
            if isinstance(value, (list, tuple)):
                if doc_value not in value:
                    return False
            elif isinstance(value, dict) and "operator" in value:
                # Support for complex filters like {"operator": "gte", "value": 5}
                operator = value["operator"]
                filter_value = value["value"]

                if operator == "gte" and doc_value < filter_value:
                    return False
                elif operator == "lte" and doc_value > filter_value:
                    return False
                elif operator == "gt" and doc_value <= filter_value:
                    return False
                elif operator == "lt" and doc_value >= filter_value:
                    return False
                elif operator == "eq" and doc_value != filter_value:
                    return False
                elif operator == "ne" and doc_value == filter_value:
                    return False
            else:
                # Simple equality check
                if doc_value != value:
                    return False

        return True

    def __len__(self) -> int:
        """Return number of documents in the store"""
        return len(self.documents)

    def __contains__(self, doc_id: str) -> bool:
        """Check if document ID exists in the store"""
        return doc_id in self.documents

    def __repr__(self) -> str:
        """String representation of the vector store"""
        return f"VectorStore(documents={len(self.documents)}, threshold={self.similarity_threshold})"