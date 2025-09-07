import os
from typing import Dict, Any


class Config:
    """Configuration management for the AI Evolution framework"""

    def __init__(self):
        self.llm_config = {
            "model_name": os.getenv("AI_MODEL_NAME", "claude-3-haiku"),
            "api_key": os.getenv("AI_API_KEY"),
            "temperature": float(os.getenv("AI_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("AI_MAX_TOKENS", "1000"))
        }

        self.rag_config = {
            "embedding_dimension": int(os.getenv("EMBEDDING_DIMENSION", "384")),
            "retrieval_count": int(os.getenv("RETRIEVAL_COUNT", "3")),
            "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        }

        self.agent_config = {
            "max_iterations": int(os.getenv("MAX_AGENT_ITERATIONS", "10")),
            "timeout_seconds": int(os.getenv("AGENT_TIMEOUT", "300")),
            "memory_limit": int(os.getenv("MEMORY_LIMIT", "1000"))
        }

    def get_llm_config(self) -> Dict[str, Any]:
        return self.llm_config

    def get_rag_config(self) -> Dict[str, Any]:
        return self.rag_config

    def get_agent_config(self) -> Dict[str, Any]:
        return self.agent_config