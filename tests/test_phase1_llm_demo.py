#!/usr/bin/env python3
"""
Demo script that shows the AI Evolution Framework in action
Perfect for Medium article screenshots and copy-paste examples
"""

import sys
import os
import time


# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.phase1_llm_workflow.bedrock_llm_client import BedrockLLMClient
from src.phase1_llm_workflow.workflow_engine import WorkflowEngine


def demo_phase1_bedrock():
    """Demonstrate Phase 1: Basic Bedrock LLM Workflow"""

    print("AI Evolution Framework - Phase 1 Demo")
    print("=" * 50)
    print("Phase 1: Basic LLM Workflow with Amazon Bedrock")
    print("=" * 50)

    # Initialize Bedrock client
    print("\nInitializing Amazon Bedrock client...")
    client = BedrockLLMClient(
        model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        region_name="us-east-1"
    )
    print(f"Connected to model: {client.model_id}")

    # Test basic invocation
    print("\nTesting basic LLM invocation...")
    query = "Explain Amazon Bedrock in one sentence."
    print(f"Query: {query}")

    start_time = time.time()
    response = client.invoke(query)
    end_time = time.time()

    print(f"Response: {response.content}")
    print(f"Model: {response.model}")
    print(f"Input tokens: {response.input_tokens}")
    print(f"Output tokens: {response.output_tokens}")
    print(f"API latency: {response.latency_ms:.2f}ms")
    print(f"Total time: {(end_time - start_time) * 1000:.2f}ms")

    # Test workflow engine
    print("\nTesting workflow engine with conversation tracking...")
    workflow = WorkflowEngine(client)

    conversation_queries = [
        "What is machine learning?",
        "How is it different from traditional programming?",
        "Can you give me a practical example?"
    ]

    for i, query in enumerate(conversation_queries, 1):
        print(f"\nConversation Turn {i}:")
        print(f"User: {query}")

        response = workflow.process_query(query)
        print(f"Assistant: {response.content[:200]}...")
        print(f"Tokens: {response.input_tokens + response.output_tokens}")

    # Show usage statistics
    print("\nUsage Statistics:")
    stats = client.get_usage_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Show conversation history
    print(f"\nConversation History: {len(workflow.conversation_history)} interactions")
    for i, interaction in enumerate(workflow.conversation_history, 1):
        print(f"  {i}. {interaction['user_input'][:50]}... -> {interaction['tokens']} tokens")

    # Cost estimation
    total_input_tokens = stats['total_input_tokens']
    total_output_tokens = stats['total_output_tokens']

    # Claude Haiku pricing (approximate)
    input_cost = (total_input_tokens / 1000) * 0.00025  # $0.25 per 1K input tokens
    output_cost = (total_output_tokens / 1000) * 0.00125  # $1.25 per 1K output tokens
    total_cost = input_cost + output_cost

    print(f"\nEstimated Cost:")
    print(f"  Input cost: ${input_cost:.6f}")
    print(f"  Output cost: ${output_cost:.6f}")
    print(f"  Total cost: ${total_cost:.6f}")

    print("\nPhase 1 Demo Complete")
    print("=" * 50)


if __name__ == "__main__":
    try:
        demo_phase1_bedrock()
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Requirements:")
        print("1. AWS credentials configured")
        print("2. Bedrock model access enabled")
        print("3. Required dependencies installed")