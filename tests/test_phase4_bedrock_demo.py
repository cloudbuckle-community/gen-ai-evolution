#!/usr/bin/env python3
"""
Demo script for Phase 4: Bedrock Multi-Agent System
Shows coordinated intelligence with specialized agents
"""

import sys
import os
import asyncio

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.phase4_agentic.bedrock_orchestrator import BedrockAgentOrchestrator


async def demo_phase4_bedrock():
    """Demonstrate Phase 4: Multi-Agent coordination with Bedrock"""

    print("AI Evolution Framework - Phase 4 Demo")
    print("=" * 50)
    print("Phase 4: Multi-Agent System with Bedrock")
    print("Orchestrated Claude 3.5 Agents with Shared Memory")
    print("=" * 50)

    # Initialize orchestrator
    orchestrator = BedrockAgentOrchestrator(
        llm_model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0"
    )

    print(f"Initialized agents: {list(orchestrator.agents.keys())}")
    print(f"Bedrock model: {orchestrator.llm_client.model_id}")

    # Test queries of varying complexity
    test_queries = [
        "Hello, how can you help me today?",  # Simple - chat only
        "Calculate 25 * 67 and store the result",  # Medium - chat + tools
        "Analyze system performance metrics and provide comprehensive optimization strategies"  # Complex - all agents
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 60)

        result = await orchestrator.process_request(query)

        print(f"Agents used: {', '.join(result['agents_used'])}")
        print(f"Response: {result['final_response']}")

        if 'execution_plan' in result:
            plan = result['execution_plan']
            print(f"Execution plan: {len(plan['steps'])} steps")
            print(f"Task complexity: {plan['analysis']['complexity']:.2f}")

    # Show system status
    status = orchestrator.get_system_status()
    print(f"\nSystem Status:")
    print(f"Memory items: {status['memory_items']}")
    print(f"Total requests: {status['total_requests']}")
    print(f"Usage: {status['usage_statistics']['total_tokens']} tokens")


def run_demo_phase4_bedrock():
    """Synchronous wrapper"""
    asyncio.run(demo_phase4_bedrock())


if __name__ == "__main__":
    try:
        run_demo_phase4_bedrock()
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Requirements: AWS credentials and Bedrock access")