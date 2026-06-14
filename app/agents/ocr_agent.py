import logging
from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.tools.ocr_tool import extract_document_text

logger = logging.getLogger(__name__)

def ocr_node(state: AgentState) -> dict:
    """
    LangGraph node for OCR extraction.
    Executes a LangChain ReAct agent to run the file parsing tools and updates the shared state.
    """
    logger.info("OCR Extraction Agent activated.")
    
    file_path = state.get("document_path")
    if not file_path:
        logger.warning("OCR node called but no document path in state.")
        return {
            "messages": [AIMessage(content="[OCR Error] No document has been uploaded yet. Please upload a file first.")]
        }

    try:
        raw_text = extract_document_text(file_path)
        preview = raw_text[:1000].strip()
        if len(raw_text) > 1000:
            preview += "..."
        logger.info("OCR extraction completed successfully.")
        return {
            "extracted_context": raw_text,
            "messages": [AIMessage(content=f"[OCR] Successfully extracted text from the document. Preview:\n\n{preview}")]
        }
    except Exception as e:
        logger.error(f"OCR agent execution failed: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content=f"[OCR Error] Failed to extract text from document: {str(e)}")]
        }
