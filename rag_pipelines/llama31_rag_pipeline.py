# Modified by Claude Sonnet 4
from langchain import hub
from langchain.prompts import PromptTemplate

from utils.build_collections import load_or_create_chroma_collection
from utils.document_loader import clone_locally
from utils.embedding_config import get_chunking_config, get_embedding_model
from utils.state_graph import build_state_graph
from utils.llm_factory import create_llm_backend

"""
Code from langchain's Build a RAG App documentation
https://python.langchain.com/docs/tutorials/rag/
"""

def get_krknctl_prompt():
    """Return krknctl-specific RAG prompt template optimized for chaos engineering commands"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a krknctl assistant specialized in chaos engineering. Analyze the provided context to find the most appropriate scenario for the user's request.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Examine the context for chaos engineering scenarios that match the user's request
2. Focus on matching the target (pods, nodes, CPU, memory, network, etc.)
3. If you find a relevant scenario, respond in this exact format:

SCENARIO: scenario-name
Brief explanation of what this scenario does
krknctl run scenario-name --namespace=default

4. Use the exact scenario name as it appears in the context
5. If no clear match exists, respond: "I'm not sure which krknctl scenario fits your request"

Answer:"""
    )


def load_llama31_rag_pipeline(
    github_repo="https://github.com/krkn-chaos/website",
    repo_path="content/en/docs",
    collection_name="krkn-docs",
    persist_dir="chroma_db",
    embedding_model="qwen-small",
    chunking_strategy="default",
    llm_backend="ollama",
):
    """# NOQA
    Load the Llama 3.1 RAG pipeline with ChromaDB persistence

    Args:
        data_path: List of documents can be path to documents folder and/or a list of Urls
        collection_name: Name for ChromaDB collection
        persist_dir: Directory to persist ChromaDB data
        embedding_model: Embedding model key (from embedding_config.py) or model name
        chunking_strategy: Chunking strategy key (from embedding_config.py)
    """

    # Get chunking configuration
    chunking_config = get_chunking_config(chunking_strategy)

    print(f"Loading documents from: {github_repo}")

    all_splits = clone_locally(
        github_repo,
        repo_path,
        **chunking_config,
    )
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

    # Define prompt for question-answering based on backend
    if llm_backend == "llamacpp":
        print("Loading krknctl-specific RAG prompt template...")
        prompt = get_krknctl_prompt()
    else:
        print("Loading standard RAG prompt template...")
        prompt = hub.pull("rlm/rag-prompt")

    print(f"Initializing {llm_backend} LLM...")
    llm = create_llm_backend(llm_backend)

    print("Building state graph...")
    graph = build_state_graph(vector_store, prompt, llm)
    print("RAG pipeline ready!")
    return graph
