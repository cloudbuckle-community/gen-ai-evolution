from typing import List, Dict, Any
from .vector_store import VectorStore, Document
from .embeddings import EmbeddingsModel
from ..phase1_llm_workflow.llm_client import LLMClient, LLMResponse


class RAGSystem:
    """Complete RAG implementation with retrieval and generation"""

    def __init__(self, llm_client: LLMClient, embeddings_model: EmbeddingsModel):
        self.llm = llm_client
        self.embeddings = embeddings_model
        self.vector_store = VectorStore()
        self.retrieval_count = 3

    def load_knowledge_base(self, documents: List[Dict[str, Any]]):
        """Load documents into the RAG system"""
        docs = []
        for i, doc_data in enumerate(documents):
            content = doc_data.get("content", "")
            metadata = doc_data.get("metadata", {})

            embedding = self.embeddings.embed_text(content)
            doc = Document(
                id=f"doc_{i}",
                content=content,
                metadata=metadata,
                embedding=embedding
            )
            docs.append(doc)

        self.vector_store.add_documents(docs)

    def query(self, question: str) -> Dict[str, Any]:
        """Process a query using RAG"""
        # Generate embedding for the question
        query_embedding = self.embeddings.embed_text(question)

        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(
            query_embedding, k=self.retrieval_count
        )

        # Build context from retrieved documents
        context = self._build_context(relevant_docs)

        # Generate response using LLM with context
        prompt = self._build_rag_prompt(question, context)
        response = self.llm.invoke(prompt)

        return {
            "response": response,
            "retrieved_documents": [
                {"content": doc.content, "metadata": doc.metadata}
                for doc in relevant_docs
            ],
            "context_length": len(context)
        }

    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        for doc in documents:
            context_parts.append(f"Document: {doc.content}")
        return "\n\n".join(context_parts)

    def _build_rag_prompt(self, question: str, context: str) -> str:
        """Build the final prompt for RAG"""
        return f"""Use the following context to answer the user's question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Answer:"""
