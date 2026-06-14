import os
import uuid
import logging
from typing import List, Tuple
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/uploads"))

# Ensure uploads folder exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

def validate_file(filename: str, file_content: bytes) -> Tuple[bool, str]:
    """
    Validates file extension, size, and header magic bytes.
    """
    # 1. Size Check
    if len(file_content) > MAX_FILE_SIZE:
        return False, "File exceeds maximum size of 10MB."

    # 2. Extension Check
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

    # 3. Magic Bytes Check
    if ext == ".pdf":
        if not file_content.startswith(b"%PDF"):
            return False, "File header mismatch. Not a valid PDF file."
    elif ext == ".png":
        if not file_content.startswith(b"\x89PNG\r\n\x1a\n"):
            return False, "File header mismatch. Not a valid PNG file."
    elif ext in {".jpg", ".jpeg"}:
        if not file_content.startswith(b"\xff\xd8\xff"):
            return False, "File header mismatch. Not a valid JPEG file."

    return True, ""

def save_uploaded_file(filename: str, file_content: bytes) -> str:
    """
    Renames the file to an unpredictable random UUID and saves it to the secure upload directory.
    Returns the absolute path of the saved file.
    """
    # Clean the filename to extract its clean extension
    safe_basename = os.path.basename(filename)
    ext = os.path.splitext(safe_basename.lower())[1]
    
    # Generate unique filename to prevent overwrites or traversal injections
    random_filename = f"{uuid.uuid4()}{ext}"
    
    # Resolve the destination path securely
    target_path = os.path.abspath(os.path.join(UPLOAD_DIR, random_filename))
    
    # Verify the path is strictly within UPLOAD_DIR
    if not target_path.startswith(UPLOAD_DIR + os.path.sep):
        raise ValueError("Security violation: Attempted path traversal out of sandbox.")

    with open(target_path, "wb") as f:
        f.write(file_content)

    logger.info(f"Saved uploaded file {filename} securely as {random_filename}")
    return target_path

def parse_pdf(file_path: str) -> str:
    """
    Extracts text from PDF using PyMuPDF (fitz).
    """
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error parsing PDF file: {e}")
        raise ValueError(f"Failed to read PDF content: {str(e)}")

def parse_image(file_path: str) -> str:
    """
    Extracts text from images using pytesseract.
    """
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract-OCR engine is not installed on the system.")
        raise ValueError("Tesseract OCR is not installed or configured on the system.")
    except Exception as e:
        logger.error(f"Error parsing Image file: {e}")
        raise ValueError(f"Failed to extract text from image: {str(e)}")

def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from a file based on its file extension.
    """
    ext = os.path.splitext(file_path.lower())[1]
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext in {".png", ".jpg", ".jpeg"}:
        return parse_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_text(text: str, filename: str) -> List[Document]:
    """
    Splits text into chunks for vector search.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": os.path.basename(filename),
                    "chunk_id": i
                }
            )
        )
    return documents
