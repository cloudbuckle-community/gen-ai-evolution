from typing import List, Dict, Any, Union
from .llm_client import LLMClient, LLMResponse

try:
    from .bedrock_llm_client import BedrockLLMClient, BedrockLLMResponse
except ImportError:
    BedrockLLMClient = None
    BedrockLLMResponse = None


class WorkflowEngine:
    """Manages LLM workflows with basic prompt engineering - supports both regular and Bedrock clients"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.conversation_history: List[Dict[str, Any]] = []

    def process_query(self, user_input: str, context: str = "") -> Union[LLMResponse, 'BedrockLLMResponse']:
        """Process a user query through the LLM workflow"""
        prompt = self._build_prompt(user_input, context)
        response = self.llm.invoke(prompt)

        self._log_interaction(user_input, response)
        return response

    def _build_prompt(self, user_input: str, context: str = "") -> str:
        """Build a structured prompt from user input and context"""
        base_prompt = "You are a helpful AI assistant. Please provide accurate and helpful responses."

        if context:
            prompt = f"{base_prompt}\n\nContext: {context}\n\nUser Question: {user_input}"
        else:
            prompt = f"{base_prompt}\n\nUser Question: {user_input}"

        return prompt

    def _log_interaction(self, user_input: str, response):
        """Log the interaction for debugging and analysis - handles both response types"""

        # Handle different response types
        if hasattr(response, 'input_tokens') and hasattr(response, 'output_tokens'):
            # Bedrock response
            total_tokens = response.input_tokens + response.output_tokens
            model_name = response.model
            latency = response.latency_ms
        elif hasattr(response, 'tokens_used'):
            # Original response
            total_tokens = response.tokens_used
            model_name = response.model
            latency = response.latency_ms
        else:
            # Fallback for unknown response types
            total_tokens = 0
            model_name = "unknown"
            latency = 0

        self.conversation_history.append({
            "timestamp": getattr(response, 'timestamp', 0),
            "user_input": user_input,
            "response": response.content,
            "model": model_name,
            "tokens": total_tokens,
            "latency_ms": latency
        })