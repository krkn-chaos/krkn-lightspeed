import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Assisted by Claude Sonnet 4

def load_and_split_docs(folder_path="data", chunk_size=1000, chunk_overlap=200, supported_extensions=None): 
    """
    Load and split documents from a folder
    
    Args:
        folder_path: Path to folder containing documents
        chunk_size: Maximum size of each text chunk
        chunk_overlap: Number of characters to overlap between chunks
        supported_extensions: List of file extensions to process (default: [".pdf", ".md"])
    """
    if supported_extensions is None:
        supported_extensions = [".pdf", ".md"]
    
    docs = []
    file_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(folder_path)
        for f in files
        if any(f.lower().endswith(ext) for ext in supported_extensions)
    ]
    
    print(f"Found {len(file_paths)} files with extensions {supported_extensions}")
    
    for path in file_paths: 
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            elif ext == ".md": 
                loader = UnstructuredMarkdownLoader(path)
                docs.extend(loader.load())
            print(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    print(f"Loaded {len(docs)} documents, splitting with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)
