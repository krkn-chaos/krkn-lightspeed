from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
#from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import Ollama
from rag_pipelines.granite_rag_pipeline import load_granite_rag_pipline
from rag_pipelines.llama31_rag_pipeline import load_llama31_rag_pipeline
from rag_pipelines.llama27_rag_pipeline import load_llama27_rag_pipeline


# UNCOMMENT THE CODE FOR THE MODEL THAT YOU ARE NOT USING BEFORE RUNNING

#START OF LLAMA 3.1 MODEL LOGIC
#llama 3.1
graph = load_llama31_rag_pipeline()

# run in a loop
while True:
    q = input("Ask a question (or type 'exit'): ")
    if q.lower() in ["exit", "quit"]:
        break
    start_time = time.time()
    result = graph.invoke({"question": q})
    end_time = time.time()
    duration = end_time - start_time
    
    print("Time taken:", round(duration, 2), "seconds")


    print("\nAnswer:", result["answer"])
#END OF LLAMA 3.1 MODEL LOGIC


'''#START OF LLAMA 2.7 MODEL LOGIC
#llama 2.7
graph = load_llama27_rag_pipeline()

# run in a loop
while True:
    q = input("Ask a question (or type 'exit'): ")
    if q.lower() in ["exit", "quit"]:
        break
    start_time = time.time()
    result = graph.invoke({"question": q})
    end_time = time.time()
    duration = end_time - start_time
    
    print("Time taken:", round(duration, 2), "seconds")


    print("\nAnswer:", result["answer"])
#END OF LLAMA 2.7 MODEL LOGIC'''


'''#START OF GRANITE MODEL LOGIC
#granite
graph = load_granite_rag_pipline()

# run in a loop
while True:
    q = input("Ask a question (or type 'exit'): ")
    if q.lower() in ["exit", "quit"]:
        break
    start_time = time.time()
    result = graph.invoke({"question": q})
    end_time = time.time()
    duration = end_time - start_time
    
    print("Time taken:", round(duration, 2), "seconds")


    print("\nAnswer:", result["answer"])
#END OF GRANITE MODEL LOGIC'''