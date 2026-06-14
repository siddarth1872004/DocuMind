import logging
from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.supervisor import supervisor_node
from app.agents.ocr_agent import ocr_node
from app.agents.rag_agent import rag_node
from app.agents.structured import structured_node

logger = logging.getLogger(__name__)

def route_next(state: AgentState) -> str:
    """
    Conditional routing function logic.
    Decides the next node to execute based on the next_agent field populated by the Supervisor.
    """
    next_agent = state.get("next_agent")
    logger.info(f"Conditional routing evaluated. State next_agent: {next_agent}")
    
    if next_agent == "ocr_agent":
        return "ocr_worker"
    elif next_agent == "rag_agent":
        return "rag_worker"
    elif next_agent == "structured_agent":
        return "structured_worker"
    else:
        # Routes to the graph endpoint to finish execution
        return END

def create_graph():
    """
    Constructs and compiles the multi-agent LangGraph workflow.
    """
    # Initialize the state graph with our AgentState structure
    workflow = StateGraph(AgentState)
    
    # 1. Add all agent nodes to the graph
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("ocr_worker", ocr_node)
    workflow.add_node("rag_worker", rag_node)
    workflow.add_node("structured_worker", structured_node)
    
    # 2. Add structural flow connections (Edges)
    # Specialist worker nodes always return control back to the Supervisor
    # to evaluate the next action or respond to the user.
    workflow.add_edge("ocr_worker", "supervisor")
    workflow.add_edge("rag_worker", "supervisor")
    workflow.add_edge("structured_worker", "supervisor")
    
    # The Supervisor node makes the dynamic routing choice
    workflow.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "ocr_worker": "ocr_worker",
            "rag_worker": "rag_worker",
            "structured_worker": "structured_worker",
            END: END
        }
    )
    
    # The graph execution always begins at the Supervisor Agent
    workflow.set_entry_point("supervisor")
    
    logger.info("LangGraph multi-agent flow successfully constructed.")
    return workflow.compile()

# Instantiated compiled graph ready to execute queries
app_graph = create_graph()
