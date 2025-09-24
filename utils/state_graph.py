import re
import time

from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


def build_state_graph(vector_store, prompt, llm):

    # Define state for application

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(
            state["question"], k=1
        )
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(
            doc.page_content for doc in state["context"]
        )
        messages = prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )
        response = llm.invoke(messages)
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

        sources_md_lines =get_context(result)
        sources_md = "\n\n".join(sources_md_lines)
        # Print sources and brief context for transparency
        print(sources_md)

# Generated using Cursor
def get_context(result):
    sources_md_lines = []
    context_docs = result.get("context", [])
    if context_docs:
        sources_md_lines.append("\nSources and context:")
        for idx, doc in enumerate(context_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            source = get_url_from_source(source)
            page = doc.metadata.get("page")
            location = f" (page {page})" if page is not None else ""
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300].rstrip() + "..."
            sources_md_lines.append(f"  {idx}. {source}{location}")
            sources_md_lines.append(f"     {snippet}")
    return sources_md_lines


def get_url_from_source(source):
    if "docs" in source:
        source = re.sub("[A-Za-z_0-9/]*content/en/docs", "https://krkn-chaos.dev/docs", source)
        source = re.sub(".md", "", source)
        source = re.sub("/_index", "", source)
    return source

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str