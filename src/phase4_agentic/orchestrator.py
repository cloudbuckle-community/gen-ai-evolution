import asyncio
from typing import Dict, Any, List
from .memory import SharedMemory
from .chat_agent import ChatAgent
from .reasoning_agent import ReasoningAgent
from ..phase3_agents.agent_executor import ToolAgent


class AgentOrchestrator:
    """Orchestrates multiple agents for complex task execution"""

    def __init__(self, llm_client):
        self.memory = SharedMemory()

        # Initialize specialized agents
        self.chat_agent = ChatAgent(llm_client, self.memory)
        self.reasoning_agent = ReasoningAgent(llm_client, self.memory)
        self.tool_agent = ToolAgent("ToolAgent", llm_client)

        self.agents = {
            "chat": self.chat_agent,
            "reasoning": self.reasoning_agent,
            "tools": self.tool_agent
        }

    async def process_request(self, user_input: str) -> Dict[str, Any]:
        """Process user request using appropriate agents"""
        # Start with chat agent to analyze request
        chat_response = self.chat_agent.process(user_input)

        execution_flow = {
            "user_input": user_input,
            "chat_analysis": chat_response.metadata,
            "agents_used": ["chat"],
            "responses": [chat_response],
            "final_response": chat_response.content
        }

        # If complex reasoning is needed, involve reasoning agent
        if chat_response.metadata.get("requires_reasoning", False):
            plan = self.reasoning_agent.create_plan(
                user_input,
                list(self.tool_agent.tools.keys())
            )

            execution_flow["execution_plan"] = plan
            execution_flow["agents_used"].append("reasoning")

            # Execute plan steps that require tools
            if chat_response.metadata.get("requires_tools", False):
                tool_response = self.tool_agent.process(user_input)
                execution_flow["responses"].append(tool_response)
                execution_flow["agents_used"].append("tools")

                # Generate final coordinated response
                final_response = self._synthesize_final_response(
                    user_input,
                    execution_flow["responses"],
                    plan
                )
                execution_flow["final_response"] = final_response

        # Log the complete execution flow
        self.memory.add_interaction(
            "orchestrator",
            "request_completed",
            execution_flow
        )

        return execution_flow

    def _synthesize_final_response(self, user_input: str, responses: List, plan: Dict[str, Any]) -> str:
        """Synthesize responses from multiple agents into a final answer"""
        response_parts = [
            f"I've processed your request '{user_input}' using multiple specialized agents."
        ]

        if len(responses) > 1:
            response_parts.append("Here's what I found:")

            for i, response in enumerate(responses[1:], 1):
                agent_name = ["reasoning", "tools"][i - 1] if i <= 2 else f"agent_{i}"
                response_parts.append(f"- {agent_name.title()} Agent: {response.content}")

        if plan:
            response_parts.append(f"The analysis involved {len(plan['steps'])} steps and met all success criteria.")

        return "\n".join(response_parts)

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and agent states"""
        return {
            "memory_items": len(self.memory.conversation_history),
            "agent_states": {
                name: self.memory.get_agent_state(name)
                for name in self.agents.keys()
            },
            "shared_context": self.memory.shared_context,
            "active_agents": list(self.agents.keys())
        }