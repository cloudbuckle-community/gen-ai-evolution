import unittest
import numpy as np
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.phase2_rag.vector_store import VectorStore, Document
from src.phase2_rag.embeddings import EmbeddingsModel
from src.phase2_rag.rag_system import RAGSystem
from src.phase1_llm_workflow.llm_client import LLMClient


class TestPhase2RAG(unittest.TestCase):
    """Test suite for Phase 2: RAG System"""

    def setUp(self):
        """Set up test fixtures"""
        self.embeddings_model = EmbeddingsModel(dimension=384)
        self.vector_store = VectorStore()
        self.llm_client = LLMClient()
        self.rag_system = RAGSystem(self.llm_client, self.embeddings_model)

        # Create test documents
        self.test_documents = [
            {
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "metadata": {"category": "programming", "language": "python"}
            },
            {
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
                "metadata": {"category": "ai", "topic": "machine_learning"}
            },
            {
                "content": "Data science combines statistics, programming, and domain expertise to extract insights from data.",
                "metadata": {"category": "data_science", "skills": ["statistics", "programming"]}
            }
        ]

    def test_embeddings_generation(self):
        """Test embeddings model functionality"""
        test_text = "This is a test sentence for embedding generation."
        embedding = self.embeddings_model.embed_text(test_text)

        self.assertEqual(embedding.shape, (384,))
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)

        # Test consistency
        embedding2 = self.embeddings_model.embed_text(test_text)
        np.testing.assert_array_almost_equal(embedding, embedding2)

    def test_vector_store_operations(self):
        """Test vector store document management"""
        # Create test document with embedding
        content = "Test document content"
        embedding = self.embeddings_model.embed_text(content)
        doc = Document(id="test_doc", content=content, metadata={}, embedding=embedding)

        # Test adding document
        self.vector_store.add_document(doc)
        self.assertIn("test_doc", self.vector_store.documents)

        # Test similarity search
        query_embedding = self.embeddings_model.embed_text("Test content")
        results = self.vector_store.similarity_search(query_embedding, k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "test_doc")

    def test_rag_system_end_to_end(self):
        """Test complete RAG system functionality"""
        # Load knowledge base
        self.rag_system.load_knowledge_base(self.test_documents)

        # Test queries
        test_queries = [
            "What is Python programming?",
            "Tell me about machine learning",
            "What skills are needed for data science?"
        ]

        for query in test_queries:
            result = self.rag_system.query(query)

            # Verify response structure
            self.assertIn("response", result)
            self.assertIn("retrieved_documents", result)
            self.assertIn("context_length", result)

            # Verify documents were retrieved
            self.assertGreater(len(result["retrieved_documents"]), 0)
            self.assertLessEqual(len(result["retrieved_documents"]), 3)

    def test_retrieval_relevance(self):
        """Test that retrieval returns relevant documents"""
        self.rag_system.load_knowledge_base(self.test_documents)

        # Query specifically about Python
        result = self.rag_system.query("Python programming language")
        retrieved_docs = result["retrieved_documents"]

        # Should retrieve the Python document
        python_doc_found = any(
            "python" in doc["content"].lower()
            for doc in retrieved_docs
        )
        self.assertTrue(python_doc_found)