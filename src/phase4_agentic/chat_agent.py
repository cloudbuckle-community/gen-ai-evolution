from typing import Dict, Any
from ..phase3_agents.base_agent import BaseAgent, AgentResponse
from .memory import SharedMemory
from typing import Dict, Any, List

class ChatAgent(BaseAgent):
    """Specialized agent for handling user interactions"""

    def __init__(self, llm_client, memory: SharedMemory):
        super().__init__("ChatAgent", llm_client)
        self.memory = memory

    def process(self, input_data: str) -> AgentResponse:
        """Process user input and determine response strategy"""
        # Analyze input complexity
        complexity_analysis = self._analyze_complexity(input_data)

        # Check conversation history for context
        recent_history = self.memory.get_recent_history(5)

        # Determine if this requires other agents
        requires_reasoning = complexity_analysis["complexity_score"] > 0.7
        requires_tools = self._check_tool_requirements(input_data)

        response_content = self._generate_chat_response(
            input_data,
            complexity_analysis,
            recent_history
        )

        # Log interaction
        self.memory.add_interaction(
            self.name,
            "chat_response",
            {
                "input": input_data,
                "complexity": complexity_analysis,
                "requires_reasoning": requires_reasoning,
                "requires_tools": requires_tools
            }
        )

        return AgentResponse(
            content=response_content,
            actions_taken=[],
            success=True,
            metadata={
                "requires_reasoning": requires_reasoning,
                "requires_tools": requires_tools,
                "complexity_score": complexity_analysis["complexity_score"]
            }
        )

    def _analyze_complexity(self, input_data: str) -> Dict[str, Any]:
        """Analyze the complexity of user input"""
        complexity_indicators = [
            "analyze", "compare", "evaluate", "plan", "strategy",
            "multiple", "complex", "detailed", "comprehensive"
        ]

        word_count = len(input_data.split())
        complexity_words = sum(1 for word in complexity_indicators if word in input_data.lower())

        complexity_score = min((complexity_words * 0.3) + (word_count / 50), 1.0)

        return {
            "complexity_score": complexity_score,
            "word_count": word_count,
            "complexity_indicators": complexity_words,
            "assessment": "high" if complexity_score > 0.7 else "medium" if complexity_score > 0.3 else "low"
        }

    def _check_tool_requirements(self, input_data: str) -> bool:
        """Check if the input requires tool usage"""
        tool_keywords = [
            "calculate", "compute", "search", "find", "look up",
            "current", "latest", "real-time", "data"
        ]
        return any(keyword in input_data.lower() for keyword in tool_keywords)

    def _generate_chat_response(self, input_data: str, complexity: Dict[str, Any],
                                history: List[Dict[str, Any]]) -> str:
        """Generate appropriate chat response"""
        if complexity["complexity_score"] > 0.7:
            return f"I understand you have a complex request about '{input_data[:50]}...'. Let me analyze this thoroughly and coordinate with specialized agents to provide you with a comprehensive response."
        elif complexity["complexity_score"] > 0.3:
            return f"I can help you with that. Let me process your request and gather the necessary information."
        else:
            return f"Sure, I can help with that straightforward request."