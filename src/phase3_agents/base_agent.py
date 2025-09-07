from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class AgentAction:
    tool_name: str
    parameters: Dict[str, Any]
    reasoning: str


@dataclass
class AgentResponse:
    content: str
    actions_taken: List[AgentAction]
    success: bool
    metadata: Dict[str, Any]


class BaseAgent(ABC):
    """Base class for AI agents"""

    def __init__(self, name: str, llm_client, tools: Dict[str, Any] = None):
        self.name = name
        self.llm = llm_client
        self.tools = tools or {}
        self.memory = []

    @abstractmethod
    def process(self, input_data: str) -> AgentResponse:
        """Process input and return response"""
        pass

    def add_tool(self, tool_name: str, tool_instance):
        """Add a tool to the agent's toolkit"""
        self.tools[tool_name] = tool_instance

    def _log_action(self, action: AgentAction):
        """Log agent actions for debugging"""
        self.memory.append({
            "timestamp": __import__("time").time(),
            "action": action
        })