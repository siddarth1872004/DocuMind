import os
import json
import logging
from typing import AsyncGenerator
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse, FileResponse
from langchain_core.messages import HumanMessage

from app.api.schemas import ChatRequest, UploadResponse
from app.services.parser_service import (
    validate_file, 
    save_uploaded_file, 
    extract_text_from_file, 
    chunk_text, 
    UPLOAD_DIR
)
from app.tools.vector_tool import vector_store_manager
from app.agents.graph import app_graph

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/process-document", 
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document (PDF or Image) into the system."
)
async def process_document(file: UploadFile = File(...)):
    """
    Accepts an uploaded PDF or image file, performs security/integrity checks,
    renames the file to a secure UUID, extracts text context, chunks it, and indexes
    it into the local FAISS vector store.
    """
    logger.info(f"Incoming document upload request: {file.filename}")
    
    # 1. Read file content
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read upload stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read upload file stream."
        )

    # 2. Run validations (Size, extension, magic bytes)
    is_valid, err_msg = validate_file(file.filename, content)
    if not is_valid:
        logger.warning(f"File validation failed: {err_msg}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg
        )

    # 3. Securely save file on disk outside web root
    try:
        saved_path = save_uploaded_file(file.filename, content)
        document_id = os.path.basename(saved_path)
    except Exception as e:
        logger.error(f"Failed to save document securely: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to securely store the document."
        )

    # 4. Extract text and index in FAISS vector database
    try:
        # Extract text context (PDF parser or pytesseract image OCR)
        extracted_text = extract_text_from_file(saved_path)
        
        # Chunk text
        chunks = chunk_text(extracted_text, file.filename)
        
        # Index chunks in local FAISS
        if chunks:
            vector_store_manager.add_documents(chunks)
            indexing_msg = f"Extracted {len(extracted_text)} characters, split and indexed into {len(chunks)} vector store chunks."
        else:
            indexing_msg = "Warning: Extracted document text was empty. Vector database not updated."

        return UploadResponse(
            status="success",
            filename=file.filename,
            document_id=document_id,
            message=indexing_msg
        )
    except Exception as e:
        logger.error(f"Document parsing/indexing failed: {e}", exc_info=True)
        # Attempt cleanup of saved file if parsing failed
        if os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Parsing/indexing failed: {str(e)}"
        )


@router.post(
    "/chat",
    summary="Triggers the LangGraph workflow and streams supervisor/agent execution steps back via SSE."
)
async def chat_endpoint(request: ChatRequest):
    """
    Asynchronously executes the multi-agent graph, streaming state updates, agent responses, 
    and node transitions back to the client using Server-Sent Events (SSE).
    """
    logger.info(f"Incoming agentic chat request: '{request.message}' for document ID: {request.document_id}")
    
    # 1. Resolve document path securely if provided
    resolved_path = None
    if request.document_id:
        # Prevent traversal attacks by isolating the basename
        safe_doc_id = os.path.basename(request.document_id)
        candidate_path = os.path.abspath(os.path.join(UPLOAD_DIR, safe_doc_id))
        
        # Verify boundary checking
        if candidate_path.startswith(UPLOAD_DIR + os.path.sep) and os.path.exists(candidate_path):
            resolved_path = candidate_path
            logger.info(f"Active query context associated with file: {resolved_path}")
        else:
            logger.warning(f"Requested document_id not found: {request.document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested document_id could not be resolved."
            )

    async def sse_event_generator() -> AsyncGenerator[str, None]:
        """
        Asynchronously streams graph execution steps in Server-Sent Events format.
        """
        try:
            # Construct the initial AgentState dictionary
            initial_state = {
                "messages": [HumanMessage(content=request.message)],
                "document_path": resolved_path,
                "next_agent": "supervisor",
                "extracted_context": None,
                "structured_data": None
            }

            yield f"data: {json.dumps({'event': 'node_start', 'node': 'graph', 'content': 'Starting Agentic Multi-Agent Workflow...'})}\n\n"

            # Execute the LangGraph streaming generator
            async for chunk in app_graph.astream(initial_state):
                # chunk contains a dictionary mapping the node name to the state updates returned
                for node_name, state_update in chunk.items():
                    yield f"data: {json.dumps({'event': 'node_start', 'node': node_name, 'content': f'Agent {node_name} is processing...' })}\n\n"
                    
                    # Stream the agent's textual response/thoughts
                    if "messages" in state_update and state_update["messages"]:
                        last_message = state_update["messages"][-1]
                        yield f"data: {json.dumps({'event': 'agent_message', 'node': node_name, 'content': last_message.content })}\n\n"
                    
                    # Stream structured extraction output if compiled
                    if "structured_data" in state_update and state_update["structured_data"]:
                        yield f"data: {json.dumps({'event': 'final_answer', 'node': node_name, 'content': 'Structured parsing successful.', 'data': state_update['structured_data'] })}\n\n"
                    
                    yield f"data: {json.dumps({'event': 'node_end', 'node': node_name, 'content': f'Finished processing agent: {node_name}' })}\n\n"

            yield f"data: {json.dumps({'event': 'node_end', 'node': 'graph', 'content': 'Multi-agent process execution completed successfully.'})}\n\n"
            
        except Exception as e:
            logger.error(f"Error encountered during LangGraph execution streaming: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'content': f'Internal multi-agent workflow error: {str(e)}'})}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Connection": "keep-alive"
        }
    )

@router.get("/")
async def serve_ui():
    """
    Serves the primary high-fidelity visual dashboard index.html.
    """
    ui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../static/index.html"))
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    else:
        raise HTTPException(status_code=404, detail="Web UI dashboard index.html not found.")
