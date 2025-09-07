import unittest
import asyncio

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.phase4_agentic.memory import SharedMemory
from src.phase4_agentic.chat_agent import ChatAgent
from src.phase4_agentic.reasoning_agent import ReasoningAgent
from src.phase4_agentic.orchestrator import AgentOrchestrator
from src.phase1_llm_workflow.llm_client import LLMClient


class TestPhase4Agentic(unittest.TestCase):
    """Test suite for Phase 4: Agentic AI"""

    def setUp(self):
        """Set up test fixtures"""
        self.llm_client = LLMClient()
        self.memory = SharedMemory()
        self.chat_agent = ChatAgent(self.llm_client, self.memory)
        self.reasoning_agent = ReasoningAgent(self.llm_client, self.memory)
        self.orchestrator = AgentOrchestrator(self.llm_client)

    def test_shared_memory_operations(self):
        """Test shared memory functionality"""
        # Test adding interaction
        self.memory.add_interaction(
            "test_agent",
            "test_interaction",
            {"data": "test_value"}
        )

        history = self.memory.get_recent_history(1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["agent"], "test_agent")

        # Test agent state management
        self.memory.update_agent_state("test_agent", {"status": "active"})
        state = self.memory.get_agent_state("test_agent")
        self.assertEqual(state["status"], "active")

        # Test shared context
        self.memory.set_context("current_task", "testing")
        context = self.memory.get_context("current_task")
        self.assertEqual(context, "testing")

    def test_chat_agent_complexity_analysis(self):
        """Test chat agent's complexity analysis"""
        test_cases = [
            ("Hello", "low"),
            ("Can you help me with a simple question?", "medium"),
            ("I need a comprehensive analysis of multiple complex factors", "high")
        ]

        for input_text, expected_complexity in test_cases:
            response = self.chat_agent.process(input_text)
            actual_complexity = response.metadata["complexity_score"]

            if expected_complexity == "low":
                self.assertLess(actual_complexity, 0.3)
            elif expected_complexity == "medium":
                self.assertGreaterEqual(actual_complexity, 0.3)
                self.assertLess(actual_complexity, 0.7)
            else:  # high
                self.assertGreaterEqual(actual_complexity, 0.7)

    def test_reasoning_agent_planning(self):
        """Test reasoning agent's planning capabilities"""
        task = "Analyze the performance metrics of our web application and provide optimization recommendations"
        available_tools = ["calculator", "datastore", "datetime"]

        plan = self.reasoning_agent.create_plan(task, available_tools)

        # Verify plan structure
        required_keys = ["task", "analysis", "steps", "estimated_duration", "success_criteria"]
        for key in required_keys:
            self.assertIn(key, plan)

        # Verify steps are logical
        self.assertGreater(len(plan["steps"]), 0)
        for step in plan["steps"]:
            self.assertIn("step_number", step)
            self.assertIn("action", step)
            self.assertIn("description", step)

    def test_orchestrator_coordination(self):
        """Test orchestrator's agent coordination"""
        test_inputs = [
            "What's 15 + 25?",  # Simple, should use chat + tools
            "Analyze the current market trends and provide strategic recommendations"  # Complex, should use all agents
        ]

        for input_text in test_inputs:
            # Note: In a real async environment, you'd use asyncio.run()
            # For this test, we'll simulate the async behavior
            result = asyncio.get_event_loop().run_until_complete(
                self.orchestrator.process_request(input_text)
            )

            self.assertIn("user_input", result)
            self.assertIn("chat_analysis", result)
            self.assertIn("agents_used", result)
            self.assertIn("final_response", result)

            # Verify chat agent was always used
            self.assertIn("chat", result["agents_used"])

    def test_system_status_monitoring(self):
        """Test system status and monitoring capabilities"""
        # Process a few requests to generate some activity
        asyncio.get_event_loop().run_until_complete(
            self.orchestrator.process_request("Test request 1")
        )
        asyncio.get_event_loop().run_until_complete(
            self.orchestrator.process_request("Test request 2")
        )

        status = self.orchestrator.get_system_status()

        self.assertIn("memory_items", status)
        self.assertIn("agent_states", status)
        self.assertIn("active_agents", status)

        # Should have recorded interactions
        self.assertGreater(status["memory_items"], 0)