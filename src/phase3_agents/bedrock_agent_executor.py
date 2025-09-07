# src/phase3_agents/bedrock_agent_executor.py
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .base_agent import BaseAgent, AgentAction, AgentResponse
from .tools import CalculatorTool, DateTimeTool, DataStoreTool
from ..phase1_llm_workflow.bedrock_llm_client import BedrockLLMClient


@dataclass
class BedrockAgentAction(AgentAction):
    """Enhanced agent action with Bedrock-specific metadata"""
    confidence_score: float = 0.0
    estimated_cost: float = 0.0
    tool_tokens_used: int = 0


class BedrockToolAgent(BaseAgent):
    """
    Bedrock-optimized agent that uses Claude's reasoning capabilities
    for intelligent tool selection and execution
    """

    def __init__(self,
                 name: str,
                 model_id: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                 region_name: str = "us-east-1",
                 profile_name: Optional[str] = None):

        # Initialize Bedrock LLM client
        self.bedrock_client = BedrockLLMClient(
            model_id=model_id,
            region_name=region_name,
            profile_name=profile_name
        )

        super().__init__(name, self.bedrock_client)

        # Initialize default tools
        self.add_tool("calculator", CalculatorTool())
        self.add_tool("datetime", DateTimeTool())
        self.add_tool("datastore", DataStoreTool())

        # Bedrock-specific configurations
        self.reasoning_temperature = 0.3  # Lower temperature for tool selection
        self.execution_temperature = 0.7  # Higher temperature for response generation
        self.max_reasoning_tokens = 500
        self.max_response_tokens = 1000

        # System prompts optimized for Claude
        self.tool_analysis_prompt = """You are an intelligent agent that can use tools to help users. Analyze the user's request and determine what tools are needed.

Available tools:
- calculator: For mathematical calculations and expressions
- datetime: For current date/time information
- datastore: For storing and retrieving data

Your task: Analyze the user's request and respond with a JSON object indicating what tools to use.

Response format:
{
    "needs_tools": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation of your analysis",
    "suggested_actions": [
        {
            "tool": "tool_name",
            "parameters": {"param": "value"},
            "reasoning": "why this tool is needed",
            "confidence": 0.0-1.0
        }
    ]
}"""

        self.response_synthesis_prompt = """You are a helpful assistant that has just executed tools to help a user. 
Create a natural, informative response that incorporates the tool results.

Be conversational and explain what you did to help the user.
If calculations were performed, show the work.
If data was stored or retrieved, confirm the operation.
If errors occurred, explain them clearly and suggest alternatives."""

    def process(self, input_data: str) -> AgentResponse:
        """Process input using Bedrock-optimized tool analysis and execution"""

        # Step 1: Analyze tool requirements using Claude's reasoning
        tool_analysis = self._bedrock_analyze_tools(input_data)

        actions_taken = []
        tool_results = []
        total_tokens = 0

        # Step 2: Execute tools if needed
        if tool_analysis.get("needs_tools", False):
            for tool_action_spec in tool_analysis.get("suggested_actions", []):
                action = BedrockAgentAction(
                    tool_name=tool_action_spec["tool"],
                    parameters=tool_action_spec["parameters"],
                    reasoning=tool_action_spec["reasoning"],
                    confidence_score=tool_action_spec.get("confidence", 0.0)
                )

                # Execute the tool
                tool_result = self._execute_bedrock_tool_action(action)
                tool_results.append(tool_result)
                actions_taken.append(action)

                # Track usage
                total_tokens += action.tool_tokens_used

        # Step 3: Generate final response using Claude with tool results
        response_content = self._bedrock_synthesize_response(
            input_data, actions_taken, tool_results
        )

        # Calculate estimated cost
        usage_stats = self.bedrock_client.get_usage_stats()

        return AgentResponse(
            content=response_content,
            actions_taken=actions_taken,
            success=True,
            metadata={
                "tool_analysis": tool_analysis,
                "bedrock_model": self.bedrock_client.model_id,
                "total_tokens": total_tokens + usage_stats.get("total_tokens", 0),
                "usage_stats": usage_stats
            }
        )

    def _bedrock_analyze_tools(self, input_data: str) -> Dict[str, Any]:
        """Use Claude to analyze what tools are needed"""

        analysis_prompt = f"""{self.tool_analysis_prompt}

User request: "{input_data}"

Analyze this request and respond with the JSON object:"""

        try:
            response = self.bedrock_client.invoke(
                prompt=analysis_prompt,
                max_tokens=self.max_reasoning_tokens,
                temperature=self.reasoning_temperature
            )

            # Parse Claude's JSON response
            analysis_text = response.content.strip()

            # Extract JSON from response (Claude sometimes adds explanation)
            if "```json" in analysis_text:
                json_start = analysis_text.find("```json") + 7
                json_end = analysis_text.find("```", json_start)
                analysis_text = analysis_text[json_start:json_end]
            elif "{" in analysis_text:
                json_start = analysis_text.find("{")
                json_end = analysis_text.rfind("}") + 1
                analysis_text = analysis_text[json_start:json_end]

            analysis = json.loads(analysis_text)

            # Add Bedrock-specific metadata
            analysis["bedrock_analysis_tokens"] = response.input_tokens + response.output_tokens
            analysis["analysis_latency_ms"] = response.latency_ms

            return analysis

        except (json.JSONDecodeError, Exception) as e:
            # Fallback to heuristic analysis if JSON parsing fails
            return self._fallback_tool_analysis(input_data)

    def _execute_bedrock_tool_action(self, action: BedrockAgentAction) -> Dict[str, Any]:
        """Execute tool action with Bedrock-specific tracking"""

        if action.tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{action.tool_name}' not available",
                "tool": action.tool_name
            }

        tool = self.tools[action.tool_name]
        start_time = time.time()

        try:
            result = tool.execute(**action.parameters)
            result["execution_time_ms"] = (time.time() - start_time) * 1000
            result["tool"] = action.tool_name

            # Estimate tokens used for tool operation (approximate)
            action.tool_tokens_used = len(str(action.parameters)) + len(str(result))

            self._log_action(action)
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": action.tool_name,
                "execution_time_ms": (time.time() - start_time) * 1000
            }

    def _bedrock_synthesize_response(self,
                                     input_data: str,
                                     actions: List[BedrockAgentAction],
                                     tool_results: List[Dict[str, Any]]) -> str:
        """Use Claude to synthesize final response from tool results"""

        if not actions:
            # Simple response for non-tool queries
            simple_prompt = f"Respond helpfully to this user request: {input_data}"
            response = self.bedrock_client.invoke(
                prompt=simple_prompt,
                max_tokens=self.max_response_tokens,
                temperature=self.execution_temperature
            )
            return response.content

        # Build context with tool results
        tool_context = []
        for action, result in zip(actions, tool_results):
            if result.get("success", False):
                tool_context.append(
                    f"Tool: {action.tool_name}\n"
                    f"Action: {action.reasoning}\n"
                    f"Parameters: {action.parameters}\n"
                    f"Result: {result}\n"
                )
            else:
                tool_context.append(
                    f"Tool: {action.tool_name}\n"
                    f"Action: {action.reasoning}\n"
                    f"Error: {result.get('error', 'Unknown error')}\n"
                )

        synthesis_prompt = f"""{self.response_synthesis_prompt}

User's original request: "{input_data}"

Tools executed:
{chr(10).join(tool_context)}

Please provide a helpful response that incorporates these tool results:"""

        response = self.bedrock_client.invoke(
            prompt=synthesis_prompt,
            max_tokens=self.max_response_tokens,
            temperature=self.execution_temperature
        )

        return response.content

    def _fallback_tool_analysis(self, input_data: str) -> Dict[str, Any]:
        """Fallback heuristic analysis if Bedrock JSON parsing fails"""

        suggested_actions = []

        # Mathematical operations
        if any(word in input_data.lower() for word in ["calculate", "math", "+", "-", "*", "/", "="]):
            # Try to extract mathematical expression
            import re
            math_pattern = r'[\d\+\-\*/\(\)\.\s]+'
            matches = re.findall(math_pattern, input_data)
            if matches:
                expression = max(matches, key=len).strip()
                suggested_actions.append({
                    "tool": "calculator",
                    "parameters": {"expression": expression},
                    "reasoning": "Mathematical calculation detected",
                    "confidence": 0.8
                })

        # Date/time requests
        if any(word in input_data.lower() for word in ["time", "date", "when", "now", "today"]):
            if "time" in input_data.lower():
                action_type = "current_time"
            else:
                action_type = "current_date"

            suggested_actions.append({
                "tool": "datetime",
                "parameters": {"action": action_type},
                "reasoning": "Date/time information requested",
                "confidence": 0.9
            })

        # Data storage operations
        if any(word in input_data.lower() for word in ["store", "save", "remember", "retrieve", "get"]):
            suggested_actions.append({
                "tool": "datastore",
                "parameters": {"action": "list"},
                "reasoning": "Data storage operation detected",
                "confidence": 0.7
            })

        return {
            "needs_tools": len(suggested_actions) > 0,
            "confidence": 0.6,  # Lower confidence for fallback
            "reasoning": "Fallback heuristic analysis used",
            "suggested_actions": suggested_actions,
            "fallback_used": True
        }

    def get_bedrock_stats(self) -> Dict[str, Any]:
        """Get comprehensive Bedrock usage statistics"""

        base_stats = self.bedrock_client.get_usage_stats()

        return {
            "agent_name": self.name,
            "bedrock_model": self.bedrock_client.model_id,
            "llm_usage": base_stats,
            "tools_available": list(self.tools.keys()),
            "actions_logged": len(self.memory),
            "reasoning_temperature": self.reasoning_temperature,
            "execution_temperature": self.execution_temperature
        }

    def estimate_action_cost(self, action: BedrockAgentAction) -> float:
        """Estimate cost for a specific action"""

        # Approximate token usage for tool analysis and execution
        analysis_tokens = 100  # Estimate for tool analysis
        execution_tokens = action.tool_tokens_used
        response_tokens = 200  # Estimate for response synthesis

        total_tokens = analysis_tokens + execution_tokens + response_tokens

        # Haiku pricing (approximate)
        if "haiku" in self.bedrock_client.model_id.lower():
            cost_per_1k_tokens = 0.00125  # Average of input/output
        else:
            cost_per_1k_tokens = 0.015  # Sonnet pricing

        return (total_tokens / 1000) * cost_per_1k_tokens


# Enhanced demo function for Phase 3
def demo_bedrock_agent():
    """Demonstrate Bedrock-optimized agent capabilities"""

    print("AI Evolution Framework - Phase 3 Demo")
    print("=" * 50)
    print("Phase 3: Bedrock-Optimized AI Agent")
    print("Model: Claude 3.5 Haiku with Intelligent Tool Selection")
    print("=" * 50)

    # Initialize Bedrock agent
    print("\nInitializing Bedrock agent...")
    agent = BedrockToolAgent(
        name="BedrockDemoAgent",
        model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0"
    )

    print(f"Agent initialized with tools: {list(agent.tools.keys())}")

    # Test queries that demonstrate tool selection
    test_queries = [
        "What's 15 * 234 + 67?",
        "What time is it right now?",
        "Store the value 'demo_session_2024' with key 'session_id'",
        "Calculate the compound interest on $5000 at 3.5% for 2 years",
        "What's the current date and store it as 'last_accessed'?"
    ]

    total_cost = 0

    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 60)

        start_time = time.time()
        response = agent.process(query)
        end_time = time.time()

        print(f"Agent Response: {response.content}")
        print(f"Success: {response.success}")
        print(f"Actions taken: {len(response.actions_taken)}")

        for action in response.actions_taken:
            print(f"  - {action.tool_name}: {action.reasoning} (confidence: {action.confidence_score:.2f})")

        # Show metadata
        metadata = response.metadata
        print(f"Total time: {(end_time - start_time) * 1000:.2f}ms")
        print(f"Bedrock tokens: {metadata.get('total_tokens', 0)}")

        # Cost estimation
        query_cost = sum(agent.estimate_action_cost(action) for action in response.actions_taken)
        total_cost += query_cost
        print(f"Estimated cost: ${query_cost:.6f}")

    # Final statistics
    print("\n" + "=" * 50)
    print("Bedrock Agent Performance Summary")
    print("=" * 50)

    stats = agent.get_bedrock_stats()
    print(f"Model: {stats['bedrock_model']}")
    print(f"Total requests: {stats['llm_usage']['request_count']}")
    print(f"Total tokens: {stats['llm_usage']['total_tokens']}")
    print(f"Actions executed: {stats['actions_logged']}")
    print(f"Tools available: {len(stats['tools_available'])}")
    print(f"Total estimated cost: ${total_cost:.6f}")

    print("\nPhase 3 Bedrock Agent Demo Complete")
    print("=" * 50)


if __name__ == "__main__":
    demo_bedrock_agent()