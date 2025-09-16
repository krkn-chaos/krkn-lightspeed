import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredURLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Assisted by Claude Sonnet 4


def docs_list():
    urls = [
        "https://krkn-chaos.dev/docs/",
        "https://krkn-chaos.dev/docs/krkn/",
        "https://krkn-chaos.dev/docs/krkn-hub/",
        "https://krkn-chaos.dev/docs/krknctl/",
        "https://krkn-chaos.dev/docs/installation/krkn/",
        "https://krkn-chaos.dev/docs/installation/krkn-hub/",
        "https://krkn-chaos.dev/docs/installation/krknctl/",
        "https://krkn-chaos.dev/docs/scenarios/cloud_setup/",
        "https://krkn-chaos.dev/docs/scenarios/",
        "https://krkn-chaos.dev/docs/getting-started/use-cases/",
        "https://krkn-chaos.dev/docs/getting-started/",
        "https://krkn-chaos.dev/docs/chaos-recommender/",
        "https://krkn-chaos.dev/docs/getting-started/getting-started-krkn/",
        "https://krkn-chaos.dev/docs/scenarios/pod-network-scenario/pod-network-chaos-krkn-hub/",  # NOQA
        "https://krkn-chaos.dev/docs/developers-guide/" "data",
    ]
    return urls


def load_and_split_docs(
    folder_path="data",
    supported_extensions=None,
):
    """# NOQA
    Load and split documents from a folder

    Args:
        folder_path: Path to folder containing documents
        chunk_size: Maximum size of each text chunk
        chunk_overlap: Number of characters to overlap between chunks
        supported_extensions: List of file extensions to process (default: [".pdf", ".md"])
    """

    if supported_extensions is None:
        supported_extensions = [".pdf", ".md"]

    file_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(folder_path)
        for f in files
        if any(f.lower().endswith(ext) for ext in supported_extensions)
    ]

    print(
        f"Found {len(file_paths)} files with extensions {supported_extensions}"
    )
    return file_paths


def load_and_split(file_paths, chunk_size=1000, chunk_overlap=200):
    """# NOQA
    Load and split documents from a folder and/or list of urls

    Args:
        file_paths: Path to folder containing documents or urls
        chunk_size: Maximum size of each text chunk
        chunk_overlap: Number of characters to overlap between chunks
    """
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    split_docs = text_splitter.split_documents(docs)
    return split_docs
