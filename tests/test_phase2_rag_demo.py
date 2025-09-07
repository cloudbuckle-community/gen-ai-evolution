#!/usr/bin/env python3
"""
Demo script for Phase 2: RAG System with Amazon Bedrock
Shows the evolution from basic LLM to knowledge-enhanced responses
"""

import sys
import os
import time

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.phase2_rag.bedrock_rag_system import BedrockRAGSystem


def demo_phase2_bedrock():
    """Demonstrate Phase 2: RAG System with Bedrock"""

    print("AI Evolution Framework - Phase 2 Demo")
    print("=" * 50)
    print("Phase 2: RAG System with Amazon Bedrock")
    print("Model: Claude 3.5 Haiku + Titan Embeddings v2")
    print("=" * 50)

    # Initialize Bedrock RAG system
    print("\nInitializing Bedrock RAG system...")
    rag_system = BedrockRAGSystem(
        llm_model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        embedding_model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1"
    )
    print("Connected to Claude 3.5 Haiku and Titan Embeddings v2")

    # Create a comprehensive knowledge base
    print("\nBuilding knowledge base...")
    knowledge_base = [
        {
            "content": "Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon via a single API. It provides the capabilities you need to build generative AI applications with security, privacy, and responsible AI.",
            "metadata": {"service": "bedrock", "category": "overview", "source": "aws_docs"}
        },
        {
            "content": "Amazon Titan Embed Text v2 is a text embeddings model that converts text into numerical representations. It supports text inputs up to 8,192 tokens and outputs vectors with 1,024 dimensions. The model is optimized for retrieval accuracy and supports over 100 languages.",
            "metadata": {"service": "titan", "category": "embeddings", "version": "v2"}
        },
        {
            "content": "Anthropic's Claude 3.5 Haiku is designed for speed and cost-effectiveness while maintaining high intelligence. It excels at tasks like customer support, content moderation, and data extraction. Haiku processes information quickly with lower latency compared to larger models.",
            "metadata": {"service": "claude", "model": "haiku", "category": "llm"}
        },
        {
            "content": "Retrieval Augmented Generation (RAG) combines information retrieval with language generation. It works by first retrieving relevant documents from a knowledge base using semantic similarity, then using those documents as context for the language model to generate more accurate and informed responses.",
            "metadata": {"concept": "rag", "category": "architecture", "technique": "retrieval"}
        },
        {
            "content": "Vector embeddings are numerical representations of text that capture semantic meaning. Similar concepts have similar vector representations, enabling semantic search. High-dimensional vectors (like Titan's 1,024 dimensions) can capture more nuanced relationships between concepts.",
            "metadata": {"concept": "embeddings", "category": "ml", "technique": "vectorization"}
        }
    ]

    print(f"Loading {len(knowledge_base)} documents into vector store...")
    start_time = time.time()
    rag_system.load_knowledge_base(knowledge_base)
    load_time = time.time() - start_time

    print(f"Knowledge base loaded in {load_time:.2f} seconds")
    print(f"Vector store contains {len(rag_system.vector_store.documents)} documents")

    # Show vector store statistics
    stats = rag_system.vector_store.get_stats()
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Memory usage: {stats['memory_usage_mb']} MB")

    # Test queries that demonstrate RAG capabilities
    print("\nTesting RAG queries...")
    test_queries = [
        "What is Amazon Bedrock and what models does it support?",
        "How does Titan embeddings work and what are its specifications?",
        "Explain the difference between Claude Haiku and other models",
        "What is RAG and how does it improve language model responses?",
        "Compare vector embeddings with traditional keyword search"
    ]

    total_tokens = 0
    total_cost = 0

    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)

        start_time = time.time()
        result = rag_system.query(query, max_tokens=200, temperature=0.3)
        end_time = time.time()

        # Extract response details
        response = result["response"]
        retrieved_docs = result["retrieved_documents"]

        print(f"Response: {response.content}")
        print(f"Retrieved {len(retrieved_docs)} relevant documents:")

        for j, doc in enumerate(retrieved_docs, 1):
            print(f"  {j}. {doc['content'][:100]}...")
            print(f"     Source: {doc['metadata']}")

        print(
            f"Tokens: {response.input_tokens} input + {response.output_tokens} output = {response.input_tokens + response.output_tokens} total")
        print(f"Latency: {response.latency_ms:.2f}ms")
        print(f"Total time: {(end_time - start_time) * 1000:.2f}ms")

        # Track usage
        total_tokens += response.input_tokens + response.output_tokens

    # Show final statistics
    print("\n" + "=" * 50)
    print("RAG System Performance Summary")
    print("=" * 50)

    # Get usage stats
    system_stats = rag_system.get_system_stats()

    print(f"Vector Store:")
    print(f"  Documents: {system_stats['vector_store']['document_count']}")
    print(f"  Embedding dimension: {system_stats['vector_store']['embedding_dimension']}")

    print(f"LLM Usage:")
    llm_stats = system_stats['llm_stats']
    print(f"  Requests: {llm_stats['request_count']}")
    print(f"  Input tokens: {llm_stats['total_input_tokens']}")
    print(f"  Output tokens: {llm_stats['total_output_tokens']}")
    print(f"  Total tokens: {llm_stats['total_tokens']}")

    print(f"Embeddings Usage:")
    embed_stats = system_stats['embedding_stats']
    print(f"  Embeddings generated: {embed_stats['embedding_count']}")
    print(f"  Embedding tokens: {embed_stats['total_input_tokens']}")

    # Cost estimation
    llm_input_cost = (llm_stats['total_input_tokens'] / 1000) * 0.00025  # Haiku input
    llm_output_cost = (llm_stats['total_output_tokens'] / 1000) * 0.00125  # Haiku output
    embedding_cost = (embed_stats['total_input_tokens'] / 1000) * 0.0001  # Titan embeddings
    total_cost = llm_input_cost + llm_output_cost + embedding_cost

    print(f"Estimated Costs:")
    print(f"  LLM input: ${llm_input_cost:.6f}")
    print(f"  LLM output: ${llm_output_cost:.6f}")
    print(f"  Embeddings: ${embedding_cost:.6f}")
    print(f"  Total: ${total_cost:.6f}")

    print("\nRAG vs Basic LLM Comparison:")
    print("  Basic LLM: Limited to training data, may hallucinate")
    print("  RAG System: Uses current knowledge base, provides sources")
    print("  Accuracy: Significantly improved with domain-specific knowledge")
    print("  Transparency: Shows which documents influenced the response")

    print("\nPhase 2 Demo Complete")
    print("=" * 50)


if __name__ == "__main__":
    try:
        demo_phase2_bedrock()
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Requirements:")
        print("1. AWS credentials configured")
        print("2. Bedrock model access enabled")
        print("3. Claude and Titan models accessible")