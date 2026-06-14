import logging
import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.agents.supervisor import get_llm

logger = logging.getLogger(__name__)

# Default schema representing parsed document content
class DefaultStructuredDocument(BaseModel):
    document_type: str = Field(
        description="The category or type of the document, e.g., Invoice, Resume, Agreement, Article, Receipt, Report, Unknown."
    )
    summary: str = Field(
        description="A concise summary of the document contents (2-3 sentences)."
    )
    extracted_metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Extracted key-value pairs such as: invoice_id, dates, names, email, total_amount, phone, organization."
    )
    key_takeaways: List[str] = Field(
        default_factory=list,
        description="Bullet points representing the most important details or items."
    )

STRUCTURED_PROMPT = """You are the Structured Output Agent of a Document Intelligence System.
Your job is to read raw document context and format it into a strict JSON object based on a Pydantic schema.

Context:
---
{context}
---

Extract the document type, summary, metadata attributes (dates, names, values, identifiers), and key takeaways from the context above.
Fill out the schema fields accurately. If a field or detail is not present in the context, do not make it up; set empty strings or lists.
"""

def structured_node(state: AgentState) -> dict:
    """
    LangGraph node for structured data extraction.
    Takes the accumulated extracted_context, parses it into the DefaultStructuredDocument schema,
    and updates the state's structured_data field.
    """
    logger.info("Structured Output Agent activated.")
    
    context = state.get("extracted_context") or ""
    if not context.strip():
        logger.warning("Structured node called but extracted_context is empty.")
        return {
            "messages": [AIMessage(content="[Structured Agent Error] No text context has been extracted yet. Please run OCR or RAG first.")]
        }

    try:
        llm = get_llm()
        # Bind Pydantic model for structured output
        structured_llm = llm.with_structured_output(DefaultStructuredDocument)
    except Exception as e:
        logger.error(f"Failed to initialize LLM for structured output: {e}", exc_info=True)
        # Safe offline fallback schema JSON
        fallback_data = {
            "document_type": "Plain Text Document (Offline)",
            "summary": "This document was parsed in offline mode without LLM intelligence.",
            "extracted_metadata": {"char_count": str(len(context))},
            "key_takeaways": ["Please verify system API keys for full LLM analysis."]
        }
        return {
            "structured_data": fallback_data,
            "messages": [AIMessage(content=f"[Structured Agent] Run fallback. Data:\n```json\n{json.dumps(fallback_data, indent=2)}\n```")]
        }

    # Setup the prompt
    prompt = ChatPromptTemplate.from_template(STRUCTURED_PROMPT)
    chain = prompt | structured_llm

    try:
        # Run structured LLM chain
        result: DefaultStructuredDocument = chain.invoke({"context": context[:30000]}) # Truncate to save token limit
        
        # Turn to dict for state serialization
        result_dict = result.model_dump()
        logger.info("Structured extraction successfully complete.")
        
        # Prepare a friendly JSON response
        json_output = json.dumps(result_dict, indent=2)
        response_message = AIMessage(
            content=f"Successfully extracted structured information from the document:\n\n```json\n{json_output}\n```"
        )
        
        # Return state updates
        return {
            "structured_data": result_dict,
            "messages": [response_message]
        }
    except Exception as e:
        logger.error(f"Structured Agent execution failed: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content=f"[Structured Agent] Failed to parse content: {str(e)}")]
        }
