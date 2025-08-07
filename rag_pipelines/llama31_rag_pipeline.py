from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from utils.document_loader import load_and_split_docs
from utils.state_graph import *

'''
Code from langchain's Build a RAG App documentation
https://python.langchain.com/docs/tutorials/rag/
'''

def load_llama31_rag_pipeline(): 
    all_splits = load_and_split_docs("data")

    # embed and store in vector database
    embedding_model = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
    vector_store = Chroma.from_documents(documents=all_splits, embedding=embedding_model)

    # Define prompt for question-answering
    # N.B. for non-US LangSmith endpoints, you may need to specify
    # api_url="https://api.smith.langchain.com" in hub.pull.
    prompt = hub.pull("rlm/rag-prompt")


    llm = Ollama(model="llama3.1", base_url="http://127.0.0.1:11434")
    
    graph = build_state_graph(vector_store, prompt, llm)
    return graph
