import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    timestamp: float


class LLMClient:
    """Simple LLM client for basic workflow operations"""

    def __init__(self, model_name: str = "claude-3-haiku", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.request_count = 0

    def invoke(self, prompt: str, temperature: float = 0.7) -> LLMResponse:
        """Invoke the LLM with a simple prompt"""
        start_time = time.time()
        self.request_count += 1

        # Simulate API call - in production, replace with actual LLM API
        response_content = self._simulate_llm_response(prompt)

        latency = (time.time() - start_time) * 1000

        return LLMResponse(
            content=response_content,
            model=self.model_name,
            tokens_used=len(prompt.split()) + len(response_content.split()),
            latency_ms=latency,
            timestamp=time.time()
        )

    def _simulate_llm_response(self, prompt: str) -> str:
        """Simulate LLM response for testing purposes"""
        if "weather" in prompt.lower():
            return "I don't have access to real-time weather data. Please check a weather service."
        elif "calculate" in prompt.lower():
            return "For accurate calculations, please use a calculator or specify the exact computation."
        else:
            return f"Based on your question about '{prompt[:50]}...', here's a general response."