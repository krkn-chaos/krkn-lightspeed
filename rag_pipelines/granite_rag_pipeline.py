import torch
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import List, TypedDict

from utils.document_loader import load_and_split

"""
Code from langchain's Build a RAG App documentation
https://python.langchain.com/docs/tutorials/rag/
"""


def load_granite_rag_pipline():
    # load and chunk contents of thepytohnPDF
    urls = [
        "https://krkn-chaos.dev/docs/",
        "https://krkn-chaos.dev/docs/krkn/",
        "https://krkn-chaos.dev/docs/krkn-hub/",
        "data",
    ]
    all_splits = load_and_split(urls)

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
        docs_content = "\n\n".join(
            doc.page_content for doc in state["context"]
        )
        messages = prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )

        # convert ChatPromptValue to plain text prompt
        if hasattr(messages, "to_messages"):  # it's a ChatPromptValue
            chat_messages = messages.to_messages()
            prompt_str = "\n".join([m.content for m in chat_messages])
        else:
            raise ValueError("Unexpected message format")

        # tokenize and run the Granite model
        input_tokens = tokenizer(prompt_str, return_tensors="pt").to(
            model.device
        )
        output_tokens = model.generate(**input_tokens, max_new_tokens=512)
        response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        return {"answer": response}

    # Compile the graph
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph
