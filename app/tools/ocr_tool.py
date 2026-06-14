import logging
from langchain_core.tools import tool
from app.services.parser_service import extract_text_from_file

logger = logging.getLogger(__name__)

@tool("extract_document_text")
def extract_document_text(file_path: str) -> str:
    """
    Extracts text from a document located at the given file_path.
    Supports PDF files and image formats (PNG, JPG, JPEG).
    Use this tool to read/parse raw document contents.
    
    Args:
        file_path (str): The absolute path to the document file.
        
    Returns:
        str: Extracted text or an error message.
    """
    logger.info(f"Tool extract_document_text invoked on file: {file_path}")
    try:
        text = extract_text_from_file(file_path)
        if not text.strip():
            return "No text could be extracted from this document."
        return f"Successfully extracted text from document:\n\n{text[:50000]}" # Limit size to prevent token limit issues
    except Exception as e:
        logger.error(f"Error executing extract_document_text tool: {e}")
        return f"Error extracting text: {str(e)}"
