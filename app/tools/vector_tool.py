import logging
from langchain_core.tools import tool
from app.services.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# Single instance of vector store manager for the application
vector_store_manager = VectorStoreManager()

@tool("search_document_database")
def search_document_database(query: str) -> str:
    """
    Queries the vector database to find contextually relevant information about the uploaded documents.
    Use this tool to find facts, figures, or answers to questions within previously uploaded documents.
    
    Args:
        query (str): The search query or question to retrieve matching passages for.
        
    Returns:
        str: Relevant passages extracted from the documents.
    """
    logger.info(f"Tool search_document_database invoked with query: {query}")
    try:
        results = vector_store_manager.similarity_search(query, k=4)
        if not results:
            return "No matching context found in the vector database."
        
        formatted_results = []
        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "unknown")
            formatted_results.append(
                f"[Doc {i+1} | Source: {source}]\n{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error executing search_document_database tool: {e}")
        return f"Error searching vector store: {str(e)}"
