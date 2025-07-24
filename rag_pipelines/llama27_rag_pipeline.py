from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

'''
Code from langchain's Build a RAG App documentation
https://python.langchain.com/docs/tutorials/rag/
'''

def load_llama27_rag_pipeline(): 
# load and chunk contents of thepytohnPDF

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

    # embed and store in vector database
    embedding_model = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
    vector_store = Chroma.from_documents(documents=all_splits, embedding=embedding_model)

    # Define prompt for question-answering
    # N.B. for non-US LangSmith endpoints, you may need to specify
    # api_url="https://api.smith.langchain.com" in hub.pull.
    prompt = hub.pull("rlm/rag-prompt")


    llm = Ollama(model="llama2:7b", base_url="http://127.0.0.1:11434")


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

    # Compile the graph
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph
