from langchain import hub
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from utils.pdf_loader import load_and_split_pdfs

'''
Code from langchain's Build a RAG App documentation
https://python.langchain.com/docs/tutorials/rag/
'''

def load_llama31_rag_pipeline(): 
# load and chunk contents of thepytohnPDF

    pdf_paths = [
        "data/pod_scenarios.pdf",
        "data/Pod-Scenarios-using-Krknctl.pdf",
        "data/Pod-Scenarios-using-Krkn-hub.pdf",
        "data/Pod-Scenarios-using-Krkn.pdf"
    ]
    all_splits = load_and_split_pdfs(pdf_paths)

    # embed and store in vector database
    embedding_model = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
    vector_store = Chroma.from_documents(documents=all_splits, embedding=embedding_model)

    # Define prompt for question-answering
    # N.B. for non-US LangSmith endpoints, you may need to specify
    # api_url="https://api.smith.langchain.com" in hub.pull.
    prompt = hub.pull("rlm/rag-prompt")


    llm = Ollama(model="llama3.1", base_url="http://127.0.0.1:11434")

    

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response}


    # llama and openaI model
        '''
        response = llm.invoke(messages)
        
        return {"answer": response}
        '''
    # Compile the graph
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph
