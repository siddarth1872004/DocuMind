# DocuMind: Document Intelligence Platform

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1-FF6B6B?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph)
[![FAISS](https://img.shields.io/badge/FAISS-vector_search-3B4CCA?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)

</div>

DocuMind is a document intelligence platform built with FastAPI, LangChain, LangGraph, and local FAISS vector search. It supports PDF/image upload, OCR extraction, semantic retrieval, and structured JSON output via a multi-agent workflow.

## Features

- Upload PDF, PNG, JPG, JPEG files and extract text using PyMuPDF or Tesseract OCR.
- Index document chunks in a local FAISS vector store.
- Multi-agent orchestration using a Supervisor, OCR Agent, RAG Retrieval Agent, and Structured Output Agent.
- Real-time streaming chat interface using Server-Sent Events (SSE).
- Offline fallback mode using fake embeddings when no API keys are configured.

## Requirements

- Python 3.14
- Docker and Docker Compose (optional)
- `GEMINI_API_KEY` or `OPENAI_API_KEY` for full LLM capabilities

## Local Setup

1. Create a Python virtual environment and activate it:

```bash
python3 -m venv test_env
source test_env/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create an optional `.env` file:

```bash
cp .env.example .env
```

4. Run the app:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

5. Open your browser at `http://localhost:8000`.

## Docker Setup

Build and run using Docker Compose:

```bash
docker compose up --build
```

The application will be available at `http://localhost:8000`.

## Environment Variables

- `GEMINI_API_KEY` - Google Gemini API key for embeddings and chat.
- `OPENAI_API_KEY` - OpenAI API key for embeddings and chat.

If neither key is configured, the app runs in offline fallback mode with fake embeddings.

## Project Structure

- `app/` - Application source code
- `app/api/` - FastAPI endpoint definitions
- `app/agents/` - Multi-agent workflow nodes
- `app/services/` - Parsing and vector store utilities
- `app/tools/` - LangChain tool wrappers
- `app/static/` - Frontend dashboard assets
- `Dockerfile` - Container image definition
- `docker-compose.yml` - Compose service configuration

## Notes

- `data/` stores uploaded files and FAISS indexes. It is ignored by git.
- `test_env/` contains the local Python environment and is ignored by git.
