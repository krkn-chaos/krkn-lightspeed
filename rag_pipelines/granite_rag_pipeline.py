import torch
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import List, TypedDict

from utils.state_graph import build_state_graph
from utils.pdf_loader import load_and_split_pdfs

"""
Code from langchain's Build a RAG App documentation
https://python.langchain.com/docs/tutorials/rag/
"""


def load_granite_rag_pipline():
    # load and chunk contents of thepytohnPDF
    pdf_paths = [
        "data/pod_scenarios.pdf",
        "data/Pod-Scenarios-using-Krknctl.pdf",
        "data/Pod-Scenarios-using-Krkn-hub.pdf",
        "data/Pod-Scenarios-using-Krkn.pdf",
    ]
    all_splits = load_and_split_pdfs(pdf_paths)

    # embed and store in vector database
    embedding_model = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B"
    )
    vector_store = Chroma.from_documents(
        documents=all_splits, embedding=embedding_model
    )

    # Define prompt for question-answering
    # N.B. for non-US LangSmith endpoints, you may need to specify
    # api_url="https://api.smith.langchain.com" in hub.pull.
    prompt = hub.pull("rlm/rag-prompt")

    # granite
    model_id = "ibm-granite/granite-3b-code-base-2k"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto" if device == "cuda" else None
    )
    model.to(device)
    model.eval()

    graph = build_state_graph(vector_store, prompt, model=model, tokenizer=tokenizer)
    return graph
