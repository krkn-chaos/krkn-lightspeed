from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredURLLoader
from langchain_pull_md.markdown_loader import PullMdLoader
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_and_split_docs(folder_path="data"):

    file_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(folder_path)
        for f in files
        if f.endswith(".pdf") or f.endswith(".md")
    ]
    
    return file_paths


def load_and_split(file_paths):
    docs = []
    urls = []
    for file in file_paths:
        if "https" in file:
            urls.append(file)
        else:
            files_in_folder = load_and_split_docs(file)
            for path in files_in_folder:
                ext = os.path.splitext(path)[1].lower()
                if ext == ".pdf":
                    loader = PyPDFLoader(path)
                elif ext == ".md": 
                    loader = UnstructuredMarkdownLoader(path)
                docs.extend(loader.load())
    if len(urls) > 0:
        loader = UnstructuredURLLoader(urls=urls, show_progress_bar=True)
        docs.extend(loader.load())

    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap = 200
    )

    split_docs = text_splitter.split_documents(docs)
    return split_docs