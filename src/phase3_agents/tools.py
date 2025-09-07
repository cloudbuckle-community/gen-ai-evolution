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


# src/phase3_agents/tools.py
from typing import Any, Dict
import json
import datetime


class CalculatorTool:
    """Simple calculator tool for agents"""

    def __init__(self):
        self.name = "calculator"

    def execute(self, expression: str) -> Dict[str, Any]:
        """Execute mathematical calculations"""
        try:
            # Safe evaluation of mathematical expressions
            allowed_chars = set("0123456789+-*/.()")
            if not all(c in allowed_chars or c.isspace() for c in expression):
                raise ValueError("Invalid characters in expression")

            result = eval(expression)
            return {
                "success": True,
                "result": result,
                "expression": expression
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "expression": expression
            }


class DateTimeTool:
    """Date and time tool for agents"""

    def __init__(self):
        self.name = "datetime"

    def execute(self, action: str) -> Dict[str, Any]:
        """Execute date/time operations"""
        now = datetime.datetime.now()

        if action == "current_time":
            return {
                "success": True,
                "result": now.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": now.timestamp()
            }
        elif action == "current_date":
            return {
                "success": True,
                "result": now.strftime("%Y-%m-%d"),
                "date": now.date().isoformat()
            }
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }


class DataStoreTool:
    """Simple data storage tool for agents"""

    def __init__(self):
        self.name = "datastore"
        self.data = {}

    def execute(self, action: str, key: str = None, value: Any = None) -> Dict[str, Any]:
        """Execute data storage operations"""
        if action == "store" and key and value is not None:
            self.data[key] = value
            return {
                "success": True,
                "action": "stored",
                "key": key,
                "value": value
            }
        elif action == "retrieve" and key:
            if key in self.data:
                return {
                    "success": True,
                    "action": "retrieved",
                    "key": key,
                    "value": self.data[key]
                }
            else:
                return {
                    "success": False,
                    "error": f"Key '{key}' not found"
                }
        elif action == "list":
            return {
                "success": True,
                "action": "listed",
                "keys": list(self.data.keys()),
                "count": len(self.data)
            }
        else:
            return {
                "success": False,
                "error": f"Invalid action or missing parameters"
            }