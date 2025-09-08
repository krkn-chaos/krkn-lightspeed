import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_docs(folder_path="data"):
    docs = []
    file_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(folder_path)
        for f in files
        if f.endswith(".pdf") or f.endswith(".md")
    ]
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(path)
            docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    return text_splitter.split_documents(docs)
