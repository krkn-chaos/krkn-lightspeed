from langchain import hub
from langchain_community.llms import Ollama

from utils.build_collections import load_or_create_chroma_collection
from utils.document_loader import load_and_split_docs
from utils.embedding_config import get_chunking_config, get_embedding_model
from utils.state_graph import build_state_graph

"""
Code from langchain's Build a RAG App documentation
https://python.langchain.com/docs/tutorials/rag/
"""


def load_llama31_rag_pipeline(
    data_path="data",
    collection_name="krkn-docs",
    persist_dir="chroma_db",
    embedding_model="qwen-small",
    chunking_strategy="default",
):
    """# NOQA
    Load the Llama 3.1 RAG pipeline with ChromaDB persistence

    Args:
        data_path: Path to documents folder
        collection_name: Name for ChromaDB collection
        persist_dir: Directory to persist ChromaDB data
        embedding_model: Embedding model key (from embedding_config.py) or model name
        chunking_strategy: Chunking strategy key (from embedding_config.py)
    """
    # Get chunking configuration
    chunking_config = get_chunking_config(chunking_strategy)

    print(f"Loading documents from: {data_path}")
    all_splits = load_and_split_docs(data_path, **chunking_config)
    print(f"Loaded and split {len(all_splits)} document chunks")

    # embed and store in vector database
    embedding_model_instance = get_embedding_model(embedding_model)

    print(f"Setting up ChromaDB collection: {collection_name}")
    vector_store = load_or_create_chroma_collection(
        collection_name=collection_name,
        embedding_model=embedding_model_instance,
        all_splits=all_splits,
        persist_dir=persist_dir,
    )

    # Define prompt for question-answering
    print("Loading RAG prompt template...")
    prompt = hub.pull("rlm/rag-prompt")

    print("Initializing Ollama LLM...")
    llm = Ollama(model="llama3.1", base_url="http://127.0.0.1:11434")

    print("Building state graph...")
    graph = build_state_graph(vector_store, prompt, llm)
    print("RAG pipeline ready!")
    return graph
