import logging
from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.tools.vector_tool import search_document_database

logger = logging.getLogger(__name__)

def rag_node(state: AgentState) -> dict:
    """
    LangGraph node for RAG retrieval.
    Executes a LangChain ReAct agent to search the FAISS vector database and returns the context.
    """
    logger.info("RAG Retrieval Agent activated.")
    
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {"messages": [AIMessage(content="[RAG Error] No user query found in conversation history.")]}

    try:
        raw_results = search_document_database(last_user_msg)
        preview = raw_results[:1200].strip()
        if len(raw_results) > 1200:
            preview += "..."
        logger.info("RAG retrieval completed successfully.")
        return {
            "extracted_context": raw_results,
            "messages": [AIMessage(content=f"[RAG] Retrieved relevant document context for the query:\n\n{preview}")]
        }
    except Exception as e:
        logger.error(f"RAG search failed: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content=f"[RAG Error] Failed to query the document database: {str(e)}")]
        }
