from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """
    Request model for the conversational interface.
    """
    message: str = Field(
        ..., 
        description="The query, prompt, or question for the multi-agent system."
    )
    document_id: Optional[str] = Field(
        None, 
        description="Optional ID of a previously uploaded document to associate with this conversation."
    )

class UploadResponse(BaseModel):
    """
    Response model for document upload and initial indexing.
    """
    status: str = Field(..., description="Status string (e.g., 'success' or 'failed').")
    filename: str = Field(..., description="The original filename uploaded by the client.")
    document_id: str = Field(..., description="The unique UUID generated to reference the saved document.")
    message: str = Field(..., description="Summary details regarding extraction and vector store indexing.")

class StreamEvent(BaseModel):
    """
    Model describing streaming execution steps passed to the client via SSE.
    """
    event: str = Field(..., description="Type of event: 'node_start', 'node_end', 'agent_message', 'final_answer', or 'error'.")
    node: Optional[str] = Field(None, description="The name of the graph node that generated this event.")
    content: str = Field(..., description="Human-readable text content associated with the event.")
    data: Optional[Dict[str, Any]] = Field(None, description="Optional structured metadata (like final JSON from Structured Output).")
