# /src/agent/graph.py
import json
from typing import List, Dict, Annotated, TypedDict
from langgraph.graph import StateGraph, END

from ..memory.semantic_memory import SemanticMemoryManager
from ..external_services.llm_client import LLMClient
from ..tools.tool_manager import ToolManager
from ..utils.logger_config import setup_logger

logger = setup_logger(__name__)

# Define the state that will be passed between nodes in the graph
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    # The final user question
    question: str
    # The context retrieved from memory
    context: str

class AgenticGraph:
    def __init__(self, llm_client: LLMClient, memory_manager: SemanticMemoryManager, tool_manager: ToolManager):
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.tool_manager = tool_manager
        self.graph = self._build_graph()

    def _build_graph(self):
        """Builds the LangGraph agentic workflow."""
        workflow = StateGraph(AgentState)

        # Define the nodes
        workflow.add_node("retrieve_context", self.retrieve_context_node)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tool_executor", self.tool_executor_node)

        # Define the edges
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_call_tool,
            {
                "call_tool": "tool_executor",
                "end": END,
            },
        )
        workflow.add_edge("tool_executor", "agent")

        return workflow.compile()

    # --- Node Functions ---

    async def retrieve_context_node(self, state: AgentState) -> Dict[str, any]:
        """Node to retrieve relevant context from the semantic memory."""
        logger.info("--- Node: Retrieving Context ---")
        question = state["messages"][-1]["content"]
        context = await self.memory_manager.get_relevant_context(question)
        logger.info(f"Retrieved context of length: {len(context) if context else 0}")
        return {"context": context or "", "question": question}

    async def agent_node(self, state: AgentState) -> Dict[str, any]:
        """The main agent node that decides what to do next."""
        logger.info("--- Node: Agent ---")
        
        system_prompt = (
            "You are a helpful research assistant. First, decide if you can answer the user's question with the provided context. "
            "If not, see if one of the available tools can help. If a tool can help, call it. "
            "If you have a tool's result, use it to formulate the final answer. "
            "Otherwise, state that you cannot answer."
        )
        
        # Prepend the context to the conversation history for the LLM
        messages = list(state["messages"]) # Make a copy
        if state.get("context"):
            messages.insert(0, {"role": "system", "content": f"CONTEXT:\n{state['context']}"})
        
        response = await self.llm_client.generate_agentic_response(
            messages=messages,
            tools=self.tool_manager.get_tool_definitions(),
            system_prompt_override=system_prompt
        )
        
        return {"messages": [self._parse_llm_response(response)]}

    async def tool_executor_node(self, state: AgentState) -> Dict[str, any]:
        """Node that executes a tool call."""
        logger.info("--- Node: Tool Executor ---")
        tool_call_message = state["messages"][-1]
        tool_call_data = json.loads(tool_call_message["content"])
        
        tool_name = tool_call_data["tool_name"]
        params = tool_call_data["parameters"]
        
        tool_result = self.tool_manager.call_tool(tool_name, **params)
        
        return {"messages": [{"role": "tool", "content": str(tool_result)}]}

    # --- Conditional Edge Logic ---

    def should_call_tool(self, state: AgentState) -> str:
        """Determines the next step after the agent node runs."""
        last_message = state["messages"][-1]
        if last_message["role"] == "tool_call":
            logger.info("Decision: Call a tool.")
            return "call_tool"
        else:
            logger.info("Decision: End graph execution.")
            return "end"

    # --- Helper Methods ---

    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """Parses the LLM's raw output into a message dictionary."""
        try:
            # Check if the response is a JSON object for a tool call
            tool_call = json.loads(response)
            if "tool_name" in tool_call and "parameters" in tool_call:
                return {"role": "tool_call", "content": response}
        except (json.JSONDecodeError, TypeError):
            # If it's not a valid JSON tool call, treat it as a final answer
            pass
        return {"role": "assistant", "content": response}

    def invoke(self, question: str):
        """Entry point to run the agentic graph."""
        initial_state = {
            "messages": [{"role": "user", "content": question}],
            "question": question,
            "context": ""
        }
        final_state = self.graph.invoke(initial_state)
        return final_state['messages'][-1]['content']
