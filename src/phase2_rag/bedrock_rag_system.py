from typing import List, Dict, Any
from .vector_store import VectorStore, Document
from .bedrock_embeddings import BedrockEmbeddingsModel
from ..phase1_llm_workflow.bedrock_llm_client import BedrockLLMClient, BedrockLLMResponse


class BedrockRAGSystem:
    """Complete RAG implementation using Amazon Bedrock"""

    def __init__(self,
                 llm_model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                 embedding_model_id: str = "amazon.titan-embed-text-v2:0",
                 region_name: str = "us-east-1",
                 profile_name: str = None):

        # Initialize Bedrock clients
        self.llm = BedrockLLMClient(
            model_id=llm_model_id,
            region_name=region_name,
            profile_name=profile_name
        )

        self.embeddings = BedrockEmbeddingsModel(
            model_id=embedding_model_id,
            region_name=region_name,
            profile_name=profile_name
        )

        # Initialize vector store
        self.vector_store = VectorStore()
        self.retrieval_count = 3

        # System prompt for RAG
        self.system_prompt = """
                    You are a helpful AI assistant. Use the provided context to answer questions accurately. 
                    If the context doesn't contain relevant information to answer the question, say so clearly. 
                    Always base your responses on the provided context when possible.
                """

    def load_knowledge_base(self, documents: List[Dict[str, Any]]):
        """Load documents into the RAG system using Bedrock embeddings"""
        print(f"Loading {len(documents)} documents into knowledge base...")

        # Extract text content
        texts = [doc.get("content", "") for doc in documents]

        # Generate embeddings using Bedrock Titan
        embeddings = self.embeddings.embed_documents(texts)

        # Create Document objects
        docs = []
        for i, (doc_data, embedding) in enumerate(zip(documents, embeddings)):
            doc = Document(
                id=f"doc_{i}",
                content=doc_data.get("content", ""),
                metadata=doc_data.get("metadata", {}),
                embedding=embedding
            )
            docs.append(doc)

        # Add to vector store
        self.vector_store.add_documents(docs)

        print(f"Successfully loaded {len(docs)} documents with embeddings")

    def query(self, question: str, max_tokens: int = 1000, temperature: float = 0.7) -> Dict[str, Any]:
        """Process a query using RAG with Bedrock models"""

        # Generate embedding for the question using Bedrock
        query_embedding = self.embeddings.embed_text(question)

        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(
            query_embedding, k=self.retrieval_count
        )

        # Build context from retrieved documents
        context = self._build_context(relevant_docs)

        # Create the prompt
        user_prompt = f"""Context:
{context}

Question: {question}

Please provide a helpful answer based on the context above."""

        # Generate response using Bedrock Claude
        response = self.llm.invoke(
            prompt=user_prompt,
            system_prompt=self.system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return {
            "response": response,
            "retrieved_documents": [
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "id": doc.id
                }
                for doc in relevant_docs
            ],
            "context_length": len(context),
            "query_embedding_dimension": len(query_embedding),
            "usage_stats": {
                "llm": self.llm.get_usage_stats(),
                "embeddings": self.embeddings.get_usage_stats()
            }
        }

    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}:\n{doc.content}")
        return "\n\n".join(context_parts)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "vector_store": {
                "document_count": len(self.vector_store.documents),
                "embedding_dimension": self.embeddings.dimension
            },
            "llm_stats": self.llm.get_usage_stats(),
            "embedding_stats": self.embeddings.get_usage_stats(),
            "retrieval_config": {
                "retrieval_count": self.retrieval_count,
                "llm_model": self.llm.model_id,
                "embedding_model": self.embeddings.model_id
            }
        }