import unittest
import time
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.phase1_llm_workflow.bedrock_llm_client import BedrockLLMClient, BedrockLLMResponse
from src.phase1_llm_workflow.workflow_engine import WorkflowEngine


class TestPhase1BedrockLLM(unittest.TestCase):
    """Test suite for Phase 1: Bedrock LLM Workflow"""

    def setUp(self):
        """Set up test fixtures"""
        self.llm_client = BedrockLLMClient(
            model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",  # Using Haiku for tests (faster/cheaper)
            region_name="us-east-1"
        )
        self.workflow_engine = WorkflowEngine(self.llm_client)

    def test_bedrock_llm_client_basic_invocation(self):
        """Test basic Bedrock LLM client functionality"""
        response = self.llm_client.invoke("What is 2+2?")

        self.assertIsInstance(response, BedrockLLMResponse)
        self.assertIsInstance(response.content, str)
        self.assertGreater(response.input_tokens, 0)
        self.assertGreater(response.output_tokens, 0)
        self.assertGreater(response.latency_ms, 0)
        self.assertEqual(response.model, "us.anthropic.claude-3-5-haiku-20241022-v1:0")

    def test_workflow_engine_with_bedrock(self):
        """Test workflow engine with Bedrock client"""
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning in one sentence",
            "What is 15 + 25?"
        ]

        for query in test_queries:
            response = self.workflow_engine.process_query(query)

            self.assertIsInstance(response, BedrockLLMResponse)
            self.assertIsInstance(response.content, str)
            self.assertGreater(len(response.content), 10)  # Should have substantial content

            # Check conversation history is maintained
            self.assertGreater(len(self.workflow_engine.conversation_history), 0)

    def test_bedrock_usage_tracking(self):
        """Test that Bedrock usage is properly tracked"""
        initial_stats = self.llm_client.get_usage_stats()

        self.llm_client.invoke("Test query for usage tracking")

        final_stats = self.llm_client.get_usage_stats()

        # Should have incremented usage
        self.assertGreater(final_stats["request_count"], initial_stats["request_count"])
        self.assertGreater(final_stats["total_input_tokens"], initial_stats["total_input_tokens"])
        self.assertGreater(final_stats["total_output_tokens"], initial_stats["total_output_tokens"])


if __name__ == '__main__':
    unittest.main()