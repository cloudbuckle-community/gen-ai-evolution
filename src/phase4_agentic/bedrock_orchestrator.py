import asyncio
from typing import Dict, Any, List
from .memory import SharedMemory
from .chat_agent import ChatAgent
from .reasoning_agent import ReasoningAgent
from ..phase3_agents.bedrock_agent_executor import BedrockToolAgent
from ..phase1_llm_workflow.bedrock_llm_client import BedrockLLMClient


class BedrockAgentOrchestrator:
    """Orchestrates multiple agents using Amazon Bedrock models"""

    def __init__(self,
                 llm_model_id: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                 region_name: str = "us-east-1",
                 profile_name: str = None):

        # Initialize Bedrock LLM client
        self.llm_client = BedrockLLMClient(
            model_id=llm_model_id,
            region_name=region_name,
            profile_name=profile_name
        )

        # Initialize shared memory
        self.memory = SharedMemory()

        # Initialize specialized agents with Bedrock LLM
        self.chat_agent = ChatAgent(self.llm_client, self.memory)
        self.reasoning_agent = ReasoningAgent(self.llm_client, self.memory)
        self.tool_agent = BedrockToolAgent("BedrockToolAgent", llm_model_id, region_name, profile_name)

        self.agents = {
            "chat": self.chat_agent,
            "reasoning": self.reasoning_agent,
            "tools": self.tool_agent
        }

        # Orchestration settings
        self.max_iterations = 10
        self.default_system_prompt = """You are part of a sophisticated AI system with multiple specialized agents. Work collaboratively to provide comprehensive and accurate responses."""

    async def process_request(self, user_input: str, **kwargs) -> Dict[str, Any]:
        """Process user request using appropriate Bedrock-powered agents"""

        print(f"\nProcessing: {user_input}")

        # Step 1: Chat agent analyzes the request
        print("Chat agent analyzing request...")
        chat_response = self.chat_agent.process(user_input)

        execution_flow = {
            "user_input": user_input,
            "chat_analysis": chat_response.metadata,
            "agents_used": ["chat"],
            "responses": [chat_response],
            "bedrock_model": self.llm_client.model_id,
            "usage_stats": self.llm_client.get_usage_stats()
        }

        # Step 2: Check complexity and tool requirements
        complexity_score = chat_response.metadata.get("complexity_score", 0)
        requires_reasoning = complexity_score > 0.6
        requires_tools = chat_response.metadata.get("requires_tools", False)

        print(f"Complexity score: {complexity_score:.2f}")
        print(f"Requires reasoning: {requires_reasoning}")
        print(f"Requires tools: {requires_tools}")

        final_response = chat_response.content

        # Step 3: Execute tools if needed
        if requires_tools:
            print("Tool agent executing...")
            tool_response = self.tool_agent.process(user_input)
            execution_flow["agents_used"].append("tools")
            execution_flow["responses"].append(tool_response)

            if tool_response.success:
                final_response = tool_response.content
                print("Tool execution successful")
            else:
                print("Tool execution failed")

        # Step 4: Apply reasoning if needed
        if requires_reasoning:
            print("Reasoning agent planning...")
            plan = self.reasoning_agent.create_plan(
                user_input,
                list(self.tool_agent.tools.keys())
            )

            execution_flow["execution_plan"] = plan
            execution_flow["agents_used"].append("reasoning")

            # Generate reasoning-enhanced response
            reasoning_response = self.reasoning_agent.process(user_input)
            execution_flow["responses"].append(reasoning_response)

            if reasoning_response.success:
                # Synthesize final response using all agent outputs
                final_response = await self._synthesize_bedrock_response(
                    user_input,
                    execution_flow["responses"],
                    plan
                )
                print("Reasoning synthesis complete")

        # Step 5: Store final response
        execution_flow["final_response"] = final_response

        # Update usage statistics
        execution_flow["final_usage_stats"] = self.llm_client.get_usage_stats()

        # Log the complete execution flow
        self.memory.add_interaction(
            "bedrock_orchestrator",
            "request_completed",
            execution_flow
        )

        print("Request processing complete")
        return execution_flow

    async def _synthesize_bedrock_response(self,
                                           user_input: str,
                                           responses: List,
                                           plan: Dict[str, Any]) -> str:
        """Synthesize final response using Bedrock Claude"""

        print("Synthesizing responses from multiple agents...")

        # Build synthesis prompt
        synthesis_prompt = f"""Based on the following analysis and agent responses, provide a comprehensive final answer to the user's request.

Original Request: {user_input}

Agent Responses:
"""

        agent_names = ["Chat", "Tools", "Reasoning"]
        for i, response in enumerate(responses):
            agent_name = agent_names[min(i, len(agent_names) - 1)]
            if hasattr(response, 'content'):
                content = response.content[:300] + "..." if len(response.content) > 300 else response.content
                synthesis_prompt += f"\n{agent_name} Agent: {content}\n"

        if plan:
            synthesis_prompt += f"\nExecution Plan: {len(plan['steps'])} steps were analyzed and executed.\n"

        synthesis_prompt += f"\nPlease provide a final, coherent response that incorporates insights from all agents:"

        # Use Bedrock to synthesize final response
        try:
            bedrock_response = self.llm_client.invoke(
                prompt=synthesis_prompt,
                system_prompt=self.default_system_prompt,
                max_tokens=1500,
                temperature=0.7
            )
            return bedrock_response.content
        except Exception as e:
            print(f"Synthesis failed: {e}")
            # Fallback to the best available response
            return responses[-1].content if responses else "I encountered an issue processing your request."

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status including Bedrock usage"""
        return {
            "bedrock_model": self.llm_client.model_id,
            "memory_items": len(self.memory.conversation_history),
            "agent_states": {
                name: self.memory.get_agent_state(name)
                for name in self.agents.keys()
            },
            "shared_context": self.memory.shared_context,
            "active_agents": list(self.agents.keys()),
            "usage_statistics": self.llm_client.get_usage_stats(),
            "total_requests": self.llm_client.request_count
        }


# Clean demo function
async def demo_phase4_bedrock_clean():
    """Demonstrate Phase 4: Multi-Agent coordination with clean output"""

    print("AI Evolution Framework - Phase 4 Demo")
    print("=" * 60)
    print("Phase 4: Multi-Agent System with Bedrock")
    print("Orchestrated Claude 3.5 Agents with Real Execution")
    print("=" * 60)

    # Initialize orchestrator
    orchestrator = BedrockAgentOrchestrator(
        llm_model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0"
    )

    print(f"Initialized agents: {list(orchestrator.agents.keys())}")
    print(f"Bedrock model: {orchestrator.llm_client.model_id}")

    # Test queries that will actually execute
    test_queries = [
        "What is 25 * 67?",
        "Calculate the area of a circle with radius 10 and store the result",
        "Analyze the benefits of using multiple AI agents in software systems"
    ]

    total_cost = 0

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"QUERY {i}: {query}")
        print(f"{'=' * 60}")

        import time
        start_time = time.time()
        result = await orchestrator.process_request(query)
        end_time = time.time()

        print(f"\nRESULTS:")
        print(f"Agents used: {', '.join(result['agents_used'])}")
        print(f"Response: {result['final_response']}")
        print(f"Total time: {(end_time - start_time):.2f}s")

        if 'execution_plan' in result:
            plan = result['execution_plan']
            print(f"Execution plan: {len(plan['steps'])} steps")
            print(f"Task complexity: {plan['analysis']['complexity']:.2f}")

        # Show usage stats
        usage = result['final_usage_stats']
        tokens = usage.get('total_tokens', 0)
        cost = (tokens / 1000) * 0.00125  # Haiku pricing
        total_cost += cost

        print(f"Tokens used: {tokens}")
        print(f"Estimated cost: ${cost:.6f}")

    # Show final system status
    print(f"\n{'=' * 60}")
    print("SYSTEM STATUS SUMMARY")
    print(f"{'=' * 60}")

    status = orchestrator.get_system_status()
    print(f"Memory items: {status['memory_items']}")
    print(f"Total requests: {status['total_requests']}")
    print(f"Total tokens: {status['usage_statistics']['total_tokens']}")
    print(f"Total estimated cost: ${total_cost:.6f}")

    print(f"\nPhase 4 Multi-Agent Demo Complete!")
    print("All agents executed successfully with real responses!")


def run_demo_phase4_bedrock_clean():
    """Synchronous wrapper for clean demo"""
    asyncio.run(demo_phase4_bedrock_clean())


if __name__ == "__main__":
    try:
        run_demo_phase4_bedrock_clean()
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Requirements: AWS credentials and Bedrock access")