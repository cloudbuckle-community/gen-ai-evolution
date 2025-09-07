import json
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentAction, AgentResponse
from .tools import CalculatorTool, DateTimeTool, DataStoreTool


class ToolAgent(BaseAgent):
    """Agent that can use tools to accomplish tasks"""

    def __init__(self, name: str, llm_client):
        super().__init__(name, llm_client)

        # Initialize default tools
        self.add_tool("calculator", CalculatorTool())
        self.add_tool("datetime", DateTimeTool())
        self.add_tool("datastore", DataStoreTool())

    def process(self, input_data: str) -> AgentResponse:
        """Process input using available tools"""
        # Analyze if tools are needed
        tool_analysis = self._analyze_tool_requirements(input_data)

        actions_taken = []

        if tool_analysis["needs_tools"]:
            for tool_action in tool_analysis["suggested_actions"]:
                action = AgentAction(
                    tool_name=tool_action["tool"],
                    parameters=tool_action["parameters"],
                    reasoning=tool_action["reasoning"]
                )

                tool_result = self._execute_tool_action(action)
                actions_taken.append(action)

                if not tool_result["success"]:
                    return AgentResponse(
                        content=f"Tool execution failed: {tool_result.get('error', 'Unknown error')}",
                        actions_taken=actions_taken,
                        success=False,
                        metadata={"tool_result": tool_result}
                    )

        # Generate final response
        response_content = self._generate_final_response(input_data, actions_taken)

        return AgentResponse(
            content=response_content,
            actions_taken=actions_taken,
            success=True,
            metadata={"tool_analysis": tool_analysis}
        )

    def _analyze_tool_requirements(self, input_data: str) -> Dict[str, Any]:
        """Analyze what tools might be needed for the input"""
        suggested_actions = []

        # Simple heuristics for tool selection
        if any(word in input_data.lower() for word in ["calculate", "math", "+", "-", "*", "/"]):
            # Extract mathematical expression
            math_terms = [word for word in input_data.split() if any(c in word for c in "0123456789+-*/")]
            if math_terms:
                suggested_actions.append({
                    "tool": "calculator",
                    "parameters": {"expression": " ".join(math_terms)},
                    "reasoning": "Mathematical calculation detected"
                })

        if any(word in input_data.lower() for word in ["time", "date", "when", "now"]):
            if "time" in input_data.lower():
                action_type = "current_time"
            else:
                action_type = "current_date"

            suggested_actions.append({
                "tool": "datetime",
                "parameters": {"action": action_type},
                "reasoning": "Date/time information requested"
            })

        if any(word in input_data.lower() for word in ["store", "save", "remember"]):
            suggested_actions.append({
                "tool": "datastore",
                "parameters": {"action": "list"},
                "reasoning": "Data storage operation detected"
            })

        return {
            "needs_tools": len(suggested_actions) > 0,
            "suggested_actions": suggested_actions
        }

    def _execute_tool_action(self, action: AgentAction) -> Dict[str, Any]:
        """Execute a tool action"""
        if action.tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{action.tool_name}' not available"
            }

        tool = self.tools[action.tool_name]
        self._log_action(action)

        return tool.execute(**action.parameters)

    def _generate_final_response(self, input_data: str, actions: List[AgentAction]) -> str:
        """Generate final response based on input and actions taken"""
        if not actions:
            return "I understand your request, but no specific tools were needed to respond."

        response_parts = ["Based on your request, I executed the following actions:"]

        for action in actions:
            response_parts.append(f"- Used {action.tool_name}: {action.reasoning}")

        return "\n".join(response_parts)