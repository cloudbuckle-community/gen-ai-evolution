from typing import Dict, Any, List
from ..phase3_agents.base_agent import BaseAgent, AgentResponse, AgentAction
from .memory import SharedMemory


class ReasoningAgent(BaseAgent):
    """Specialized agent for complex reasoning and planning"""

    def __init__(self, llm_client, memory: SharedMemory):
        super().__init__("ReasoningAgent", llm_client)
        self.memory = memory

    def create_plan(self, task: str, available_tools: List[str]) -> Dict[str, Any]:
        """Create an execution plan for complex tasks"""
        # Analyze the task
        task_analysis = self._analyze_task(task)

        # Break down into steps
        steps = self._decompose_task(task, task_analysis, available_tools)

        # Create execution plan
        plan = {
            "task": task,
            "analysis": task_analysis,
            "steps": steps,
            "estimated_duration": len(steps) * 30,  # seconds
            "success_criteria": self._define_success_criteria(task)
        }

        # Log the planning process
        self.memory.add_interaction(
            self.name,
            "plan_created",
            plan
        )

        return plan

    def process(self, input_data: str) -> AgentResponse:
        """Process complex reasoning tasks"""
        reasoning_result = self._perform_reasoning(input_data)

        return AgentResponse(
            content=reasoning_result["conclusion"],
            actions_taken=[],
            success=reasoning_result["success"],
            metadata={
                "reasoning_steps": reasoning_result["steps"],
                "confidence": reasoning_result["confidence"]
            }
        )

    def _analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze task complexity and requirements"""
        analysis = {
            "task_type": self._classify_task_type(task),
            "complexity": self._assess_complexity(task),
            "data_requirements": self._identify_data_needs(task),
            "output_format": self._determine_output_format(task)
        }
        return analysis

    def _decompose_task(self, task: str, analysis: Dict[str, Any], available_tools: List[str]) -> List[Dict[str, Any]]:
        """Break down complex task into executable steps"""
        steps = []

        # Step 1: Information gathering
        if analysis["data_requirements"]:
            steps.append({
                "step_number": 1,
                "action": "gather_information",
                "description": "Collect necessary data and information",
                "tools_needed": [tool for tool in available_tools if tool in ["web_search", "datastore"]],
                "estimated_time": 15
            })

        # Step 2: Analysis
        if analysis["complexity"] > 0.5:
            steps.append({
                "step_number": len(steps) + 1,
                "action": "analyze_data",
                "description": "Analyze collected information",
                "tools_needed": [tool for tool in available_tools if tool in ["calculator", "analysis"]],
                "estimated_time": 20
            })

        # Step 3: Synthesis
        steps.append({
            "step_number": len(steps) + 1,
            "action": "synthesize_response",
            "description": "Create final response based on analysis",
            "tools_needed": [],
            "estimated_time": 10
        })

        return steps

    def _classify_task_type(self, task: str) -> str:
        """Classify the type of task"""
        if any(word in task.lower() for word in ["analyze", "analysis"]):
            return "analysis"
        elif any(word in task.lower() for word in ["compare", "comparison"]):
            return "comparison"
        elif any(word in task.lower() for word in ["plan", "strategy"]):
            return "planning"
        else:
            return "general"

    def _assess_complexity(self, task: str) -> float:
        """Assess task complexity on a scale of 0-1"""
        complexity_factors = [
            ("multiple" in task.lower(), 0.2),
            ("complex" in task.lower(), 0.3),
            ("detailed" in task.lower(), 0.2),
            ("comprehensive" in task.lower(), 0.3),
            (len(task.split()) > 20, 0.2)
        ]

        return min(sum(weight for condition, weight in complexity_factors if condition), 1.0)

    def _identify_data_needs(self, task: str) -> List[str]:
        """Identify what data is needed for the task"""
        data_needs = []

        if any(word in task.lower() for word in ["current", "latest", "recent"]):
            data_needs.append("real_time_data")
        if any(word in task.lower() for word in ["historical", "past", "previous"]):
            data_needs.append("historical_data")
        if any(word in task.lower() for word in ["market", "business", "industry"]):
            data_needs.append("market_data")

        return data_needs

    def _determine_output_format(self, task: str) -> str:
        """Determine the best output format for the task"""
        if any(word in task.lower() for word in ["report", "document"]):
            return "structured_report"
        elif any(word in task.lower() for word in ["list", "items"]):
            return "bulleted_list"
        else:
            return "narrative"

    def _define_success_criteria(self, task: str) -> List[str]:
        """Define criteria for successful task completion"""
        return [
            "Task requirements understood and addressed",
            "All necessary information gathered",
            "Analysis completed with logical reasoning",
            "Response provided in appropriate format"
        ]

    def _perform_reasoning(self, input_data: str) -> Dict[str, Any]:
        """Perform step-by-step reasoning"""
        reasoning_steps = [
            "Understood the problem statement",
            "Identified key components and requirements",
            "Applied logical reasoning to analyze the situation",
            "Drew conclusions based on available information"
        ]

        return {
            "steps": reasoning_steps,
            "conclusion": f"Based on reasoning about '{input_data}', here are my findings...",
            "confidence": 0.85,
            "success": True
        }