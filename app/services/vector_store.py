import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import FakeEmbeddings

logger = logging.getLogger(__name__)

# Constants for storage
INDEX_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/faiss_index"))

def get_embeddings_model():
    """
    Selects and returns the embedding model based on available environment variables.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key:
        logger.info("Using Google Generative AI Embeddings.")
        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=gemini_key)
    elif openai_key:
        logger.info("Using OpenAI Embeddings.")
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
    else:
        logger.warning(
            "No GEMINI_API_KEY or OPENAI_API_KEY found. "
            "Falling back to FakeEmbeddings for testing/demonstration purposes."
        )
        return FakeEmbeddings(size=1536)

class VectorStoreManager:
    """
    Manages local FAISS vector databases for document intelligence indexing and retrieval.
    """
    def __init__(self):
        self.embeddings = get_embeddings_model()
        self.index_path = INDEX_DIRECTORY
        os.makedirs(self.index_path, exist_ok=True)
        self.db = None
        self._load_or_initialize()

    def _load_or_initialize(self):
        """
        Loads the FAISS index if it exists on disk, otherwise initializes an empty one.
        """
        try:
            if os.path.exists(os.path.join(self.index_path, "index.faiss")):
                # allow_dangerous_deserialization is required for local loading in langchain >= 0.1
                self.db = FAISS.load_local(
                    folder_path=self.index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Successfully loaded existing FAISS vector store from disk.")
            else:
                logger.info("No existing FAISS index found. Creating a new empty one.")
                # FAISS requires at least one document to initialize, so we create a dummy one
                initial_doc = [Document(page_content="System initialized.", metadata={"source": "system"})]
                self.db = FAISS.from_documents(initial_doc, self.embeddings)
                self.db.save_local(self.index_path)
        except Exception as e:
            logger.error(f"Error loading FAISS vector store: {e}. Reinitializing empty index.", exc_info=True)
            initial_doc = [Document(page_content="System re-initialized after error.", metadata={"source": "system"})]
            self.db = FAISS.from_documents(initial_doc, self.embeddings)
            self.db.save_local(self.index_path)

    def add_documents(self, documents: List[Document]):
        """
        Adds documents to the FAISS store and saves the index to disk.
        """
        if not self.db:
            self._load_or_initialize()
        
        self.db.add_documents(documents)
        self.db.save_local(self.index_path)
        logger.info(f"Added {len(documents)} document chunks to FAISS and saved index.")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs similarity search on the vector store.
        """
        if not self.db:
            self._load_or_initialize()
        
        return self.db.similarity_search(query, k=k)
