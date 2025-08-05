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
from utils.state_graph import run_question_loop


# UNCOMMENT THE CODE FOR THE MODEL THAT YOU ARE NOT USING BEFORE RUNNING


#START OF LLAMA 3.1 MODEL LOGIC
#llama 3.1
graph = load_llama31_rag_pipeline()

# run in a loop
run_question_loop(graph)

#END OF LLAMA 3.1 MODEL LOGIC


'''#START OF GRANITE MODEL LOGIC
#granite
graph = load_granite_rag_pipline()

# run in a loop
run_question_loop(graph)

#END OF GRANITE MODEL LOGIC
>>>>>>> ac1601d (called graph function in main)


#START OF LLAMA 2.7 MODEL LOGIC
#llama 2.7
graph = load_llama27_rag_pipeline()

# run in a loop
run_question_loop(graph)

#END OF LLAMA 2.7 MODEL LOGIC
'''