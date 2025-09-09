import streamlit as st

from rag_pipelines.llama31_rag_pipeline import load_llama31_rag_pipeline
from utils.document_loader import docs_list

# from granite_rag_pipeline import load_granite_rag_pipline


# Code from streamlit LLM chat documentation
# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps


st.title("krkn lightspeed RAG Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_graph():
    urls = docs_list()
    return load_llama31_rag_pipeline(urls)
    # return load_granite_rag_pipeline()


graph = get_graph()

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# React to user input
if prompt := st.chat_input("Ask about krkn pod chaos scenarios..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke({"question": prompt})
            response = result["answer"]
            st.markdown(response)
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
