from langchain import hub
from langchain_community.llms import Ollama

from utils.build_collections import load_or_create_chroma_collection
from utils.document_loader import load_and_split
from utils.embedding_config import get_chunking_config, get_embedding_model
from utils.state_graph import build_state_graph

"""
Code from langchain's Build a RAG App documentation
https://python.langchain.com/docs/tutorials/rag/
"""


def load_llama27_rag_pipeline(
    data_path=["data"],
    collection_name="krkn-docs",
    persist_dir="chroma_db",
    embedding_model="qwen-small",
    chunking_strategy="default",
):
    """# NOQA
    Load the Llama 2.7 RAG pipeline with ChromaDB persistence

    Args:
        data_path: List of documents can be path to documents folder and/or a list of Urls
        collection_name: Name for ChromaDB collection
        persist_dir: Directory to persist ChromaDB data
        embedding_model: Embedding model key (from embedding_config.py) or model name
        chunking_strategy: Chunking strategy key (from embedding_config.py)
    """
    # load and chunk contents of thepytohnPDF

    chunking_config = get_chunking_config(chunking_strategy)
    embedding_model_instance = get_embedding_model(embedding_model)
    all_splits = load_and_split(data_path, chunking_config)
    print(f"Setting up ChromaDB collection: {collection_name}")
    vector_store = load_or_create_chroma_collection(
        collection_name=collection_name,
        embedding_model=embedding_model_instance,
        all_splits=all_splits,
        persist_dir=persist_dir,
    )

    # Define prompt for question-answering
    # N.B. for non-US LangSmith endpoints, you may need to specify
    # api_url="https://api.smith.langchain.com" in hub.pull.
    print("Loading RAG prompt template...")
    prompt = hub.pull("rlm/rag-prompt")

    print("Initializing Ollama LLM...")
    llm = Ollama(model="llama2:7b", base_url="http://127.0.0.1:11434")

    print("Building state graph...")
    graph = build_state_graph(vector_store, prompt, llm)
    print("RAG pipeline ready!")
    return graph
