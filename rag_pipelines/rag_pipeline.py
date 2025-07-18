from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import Ollama

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def load_rag_pipeline():
    loader1 = PyPDFLoader("data/pod_scenarios.pdf")
    loader2 = PyPDFLoader("data/Pod-Scenarios-using-Krknctl.pdf")
    loader3 = PyPDFLoader("data/Pod-Scenarios-using-Krkn-hub.pdf")
    loader4 = PyPDFLoader("data/Pod-Scenarios-using-Krkn.pdf")

    docs1 = loader1.load()
    docs2 = loader2.load()
    docs3 = loader3.load()
    docs4 = loader4.load()


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs1 + docs2+ docs3+ docs4)

    embedding_model = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
    vector_store = Chroma.from_documents(documents=all_splits, embedding=embedding_model)

    llm = Ollama(model="llama3.1", base_url="http://127.0.0.1:11434")

    prompt = hub.pull("rlm/rag-prompt")

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph
