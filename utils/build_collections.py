from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma


def load_or_create_chroma_collection(collection_name, embedding_model, all_splits, persist_dir="chroma_db"):

    client = PersistentClient(path=persist_dir)

    try:
        client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except:
        client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}")

    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    vector_store.persist()
    return vector_store
