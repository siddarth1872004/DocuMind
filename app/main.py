import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import router as api_router

# Load environment configuration
load_dotenv()

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Verify API Keys on startup
gemini_key = os.getenv("GEMINI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
if not gemini_key and not openai_key:
    logger.warning(
        "!!! WARNING !!! Neither GEMINI_API_KEY nor OPENAI_API_KEY is defined in the environment. "
        "The system will execute in offline/fallback mock mode."
    )

def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application instance.
    """
    app = FastAPI(
        title="Agentic Document Intelligence System",
        description="A multi-agent document processor using LangGraph, LangChain, and FastAPI.",
        version="1.0.0"
    )

    # Configure secure CORS settings
    # In a production context, replace "*" with trusted domains.
    # allow_credentials is intentionally False: this API uses no cookies/sessions,
    # and combining a wildcard origin with allow_credentials=True would let any
    # origin make credentialed requests.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register API endpoints
    app.include_router(api_router)

    # Mount static dashboard folder
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
    if os.path.exists(static_dir):
        logger.info(f"Mounting static files from: {static_dir}")
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    else:
        logger.error(f"Static directory not found at {static_dir}. UI dashboard might not render.")

    logger.info("FastAPI application instance created successfully.")
    return app

app = create_app()
