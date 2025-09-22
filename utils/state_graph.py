import re
import time
import logging

from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Setup debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_state_graph(vector_store, prompt, llm):

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State):
        # For 1B model, use only the most relevant document and truncate it
        retrieved_docs = vector_store.similarity_search(
            state["question"], k=1
        )

        # Truncate document content for small models
        if retrieved_docs:
            doc = retrieved_docs[0]
            # Keep only first 800 characters to fit in small model context
            if len(doc.page_content) > 800:
                doc.page_content = doc.page_content[:800] + "..."

        # Log retrieval details for debugging
        logger.info(f"🔍 DEBUG: Retrieved {len(retrieved_docs)} documents for question")
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', 'unknown')
            preview = doc.page_content[:100].replace('\n', ' ')
            logger.info(f"   Doc {i+1}: {source} - {preview}...")
            logger.info(f"   Content length: {len(doc.page_content)} chars")

        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(
            doc.page_content for doc in state["context"]
        )

        # Debug log context preparation
        logger.info(f"🧠 DEBUG: Preparing context from {len(state['context'])} documents")
        logger.info(f"📝 DEBUG: Total context length: {len(docs_content)} characters")

        # Show a preview of the context being sent to LLM
        context_preview = docs_content[:500].replace('\n', ' ')
        logger.info(f"📜 DEBUG: Context preview: {context_preview}...")

        messages = prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )

        # Debug log prompt preparation
        if hasattr(messages, 'content'):
            prompt_preview = str(messages.content)[:300].replace('\n', ' ')
            logger.info(f"📝 DEBUG: Prompt preview: {prompt_preview}...")
        elif isinstance(messages, list) and messages:
            prompt_preview = str(messages[0])[:300].replace('\n', ' ') if messages else "Empty"
            logger.info(f"📝 DEBUG: Prompt preview: {prompt_preview}...")

        logger.info(f"🤖 DEBUG: Invoking LLM for generation...")
        response = llm.invoke(messages)

        # Debug log response
        response_preview = str(response)[:200].replace('\n', ' ')
        logger.info(f"✨ DEBUG: LLM response preview: {response_preview}...")

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
        logger.info(f"🚀 DEBUG: Starting question processing: '{q}'")
        start_time = time.time()
        result = graph.invoke({"question": q})
        end_time = time.time()
        duration = end_time - start_time

        print("Time taken:", round(duration, 2), "seconds")
        logger.info(f"⏱️ DEBUG: Question processed in {duration:.2f} seconds")
        logger.info(f"🏁 DEBUG: Retrieved {len(result.get('context', []))} documents for context")

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