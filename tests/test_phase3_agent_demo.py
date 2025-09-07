#!/usr/bin/env python3
"""
Demo script for Phase 3: Bedrock-Optimized AI Agents
Shows intelligent tool selection and execution using Claude's reasoning
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.phase3_agents.bedrock_agent_executor import demo_bedrock_agent

if __name__ == "__main__":
    try:
        demo_bedrock_agent()
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Requirements:")
        print("1. AWS credentials configured")
        print("2. Bedrock model access enabled")
        print("3. Claude models accessible")