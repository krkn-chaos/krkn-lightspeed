import time

from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


def build_state_graph(vector_store, prompt, llm=None, tokenizer=None, model=None):

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
        if llm: 
            response = llm.invoke(messages)
        else:
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


def run_question_loop(graph):
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
