from typing import Annotated, Sequence, TypedDict, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the shared state passed between the Supervisor Agent and all specialist worker agents.
    In LangGraph, each node (agent) receives this state, executes its logic, and returns updates
    to the state. The updates are merged back into the main state dictionary.
    """
    
    # The running conversation history.
    # Annotated with `add_messages` to ensure that messages returned by nodes are automatically
    # appended (concatenated) to the list rather than overwriting it.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # The router variable set by the Supervisor Agent to indicate which worker node to execute next.
    # Valid values: "ocr_agent", "rag_agent", "structured_agent", or "__end__".
    # Since nodes return a dictionary, return {"next_agent": "ocr_agent"} will update this value.
    next_agent: str
    
    # The target document's absolute file path. Passed to worker agents to allow parsing tools.
    document_path: Optional[str]
    
    # Accumulated text/context parsed from files (either via raw OCR or FAISS retrieval).
    # Specialist nodes write to this field, and subsequent nodes (like the Structured Output Agent)
    # can read from it to generate structured JSON formats.
    extracted_context: Optional[str]
    
    # Holds the final parsed/structured dictionary content produced by the Structured Output Agent.
    structured_data: Optional[Dict[str, Any]]
