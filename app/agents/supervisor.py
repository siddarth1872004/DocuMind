import os
import logging
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from app.agents.state import AgentState

logger = logging.getLogger(__name__)

# Pydantic schema for supervisor routing decisions
class RouterDecision(BaseModel):
    next_agent: Literal["ocr_agent", "rag_agent", "structured_agent", "__end__"] = Field(
        description="The next specialist agent to route to, or '__end__' if the user's task is fully resolved."
    )
    instructions: str = Field(
        description="Contextual instructions, search query, or formatting guidelines for the target specialist agent."
    )

def get_llm() -> BaseChatModel:
    """
    Retrieves the configured LLM based on environment variables.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key:
        logger.info("Initializing Gemini chat model (gemini-1.5-flash).")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=gemini_key,
            temperature=0.0
        )
    elif openai_key:
        logger.info("Initializing OpenAI chat model (gpt-4o-mini).")
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=0.0
        )
    else:
        raise ValueError(
            "Security/Configuration error: Neither GEMINI_API_KEY nor OPENAI_API_KEY is configured. "
            "Please check your docker-compose or env settings."
        )

# Define the supervisor prompt
SUPERVISOR_PROMPT = """You are the Supervisor Agent of a multi-agent Document Intelligence System.
Your job is to coordinate and delegate tasks to three specialist worker agents:
1. `ocr_agent`: Handles text extraction from raw document files (PDFs/Images). Call this if we have a document_path but have not extracted its text yet.
2. `rag_agent`: Handles semantic retrieval queries from the indexed database. Call this if the user is asking questions about the document contents and we need to retrieve relevant passages.
3. `structured_agent`: Formats text or context into strict structured JSON. Call this if the user explicitly asks for structured output, JSON extraction, or schema formatting.

Your state:
- Current document path: {document_path}
- Extracted text status: {has_extracted_context}

Analyze the user's query and the chat history carefully. Decide who to route to next. If you have all the information and the query has been answered, route to `__end__`.
"""

def supervisor_node(state: AgentState) -> dict:
    """
    LangGraph node function.
    Reads current state, invokes the LLM with structured output, and returns state updates
    specifying the 'next_agent' to route to.
    """
    logger.info("Supervisor Agent activated.")
    
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(RouterDecision)
    except Exception as e:
        logger.error(f"Failed to initialize LLM/Structured output: {e}. Falling back to default routing.", exc_info=True)
        # Safe fallback if LLM is not configured properly (helps with local startup test)
        if state.get("document_path") and not state.get("extracted_context"):
            return {
                "next_agent": "ocr_agent",
                "messages": [AIMessage(content="[Fallback Routing] Let me extract text from the file first.")]
            }
        return {
            "next_agent": "__end__",
            "messages": [AIMessage(content="Please configure GEMINI_API_KEY or OPENAI_API_KEY in the environment to enable full Agentic Routing.")]
        }

    # Setup the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        MessagesPlaceholder(variable_name="messages")
    ])

    # Format the inputs
    has_extracted = "Extracted" if state.get("extracted_context") else "Empty"
    formatted_prompt = prompt.partial(
        document_path=state.get("document_path", "None"),
        has_extracted_context=has_extracted
    )

    # Invoke LLM
    try:
        chain = formatted_prompt | structured_llm
        decision: RouterDecision = chain.invoke({"messages": state["messages"]})
        
        logger.info(f"Supervisor decided: Route to {decision.next_agent} with instructions: {decision.instructions}")
        
        # Prepare a message to inform the user about the supervisor routing choice
        routing_msg = AIMessage(
            content=f"[Supervisor] Routing task to `{decision.next_agent}`. Instructions: {decision.instructions}"
        )
        
        # Return state update dictionary. 
        # Under the hood, LangGraph merges this dict into the main AgentState.
        return {
            "next_agent": decision.next_agent,
            "messages": [routing_msg]
        }
    except Exception as e:
        logger.error(f"Supervisor LLM execution error: {e}", exc_info=True)
        # Fail safe
        return {
            "next_agent": "__end__",
            "messages": [AIMessage(content="Supervisor agent encountered an internal LLM error.")]
        }
