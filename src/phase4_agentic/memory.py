import time
from typing import Dict, Any, List, Optional
import json


class SharedMemory:
    """Shared memory system for multi-agent coordination"""

    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.shared_context: Dict[str, Any] = {}

    def add_interaction(self, agent_name: str, interaction_type: str, data: Dict[str, Any]):
        """Add an interaction to the shared memory"""
        interaction = {
            "timestamp": time.time(),
            "agent": agent_name,
            "type": interaction_type,
            "data": data
        }
        self.conversation_history.append(interaction)

    def get_agent_state(self, agent_name: str) -> Dict[str, Any]:
        """Get the current state of an agent"""
        return self.agent_states.get(agent_name, {})

    def update_agent_state(self, agent_name: str, state: Dict[str, Any]):
        """Update an agent's state"""
        self.agent_states[agent_name] = state

    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]

    def set_context(self, key: str, value: Any):
        """Set shared context information"""
        self.shared_context[key] = value

    def get_context(self, key: str) -> Any:
        """Get shared context information"""
        return self.shared_context.get(key)