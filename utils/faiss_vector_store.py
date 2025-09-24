import os
import json
import logging
from typing import List
import faiss
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class FAISSVectorStore:
    """FAISS-based vector store that mimics langchain VectorStore interface"""

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
        self.load_index()

    def load_index(self):
        """Load pre-built FAISS index"""
        logger.info(f"Loading index from {self.index_dir}")

        # Load FAISS index
        index_path = os.path.join(self.index_dir, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        self.index = faiss.read_index(index_path)

        # Load documents metadata
        docs_path = os.path.join(self.index_dir, "documents.json")
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"Documents file not found: {docs_path}")
        with open(docs_path, 'r') as f:
            self.documents = json.load(f)

        # DEBUG: Print indexed documents summary
        print(f"\n[DEBUG] INDEXED DOCUMENTS SUMMARY:")
        print(f"Total documents: {len(self.documents)}")
        doc_types = {}
        for doc in self.documents:
            doc_type = "website" if "krkn-chaos.dev" in doc.get("url", "") else "other"
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        for doc_type, count in doc_types.items():
            print(f"- {doc_type}: {count} documents")
        print(f"Sample document titles: {[doc.get('title', 'untitled')[:50] for doc in self.documents[:5]]}")
        print("[DEBUG] Index loaded successfully\n")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents and return as langchain Documents"""
        if not self.index:
            raise RuntimeError("Index not loaded")

        # Create query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, k)

        # Convert to langchain Documents
        documents = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc_data = self.documents[idx]

                # Create langchain Document with metadata
                doc = Document(
                    page_content=doc_data["content"],
                    metadata={
                        "source": doc_data["source"],
                        "title": doc_data["title"],
                        "url": doc_data["url"],
                        "relevance_score": float(score)
                    }
                )
                documents.append(doc)

        # DEBUG: Print retrieved context
        print(f"\n[DEBUG] RETRIEVED CONTEXT for query: '{query}'")
        print(f"Found {len(documents)} relevant documents:")
        for i, doc in enumerate(documents):
            print(f"- Doc {i+1}: {doc.metadata.get('title', 'untitled')[:50]} (score: {doc.metadata.get('relevance_score', 0):.3f})")
            print(f"  Content preview: {doc.page_content[:100]}...")
        print("[DEBUG] Context retrieval completed\n")

        return documents

class SimpleStateGraph:
    def __init__(self, retrieve_fn, generate_fn, vector_store_ref):
        self.retrieve = retrieve_fn
        self.generate = generate_fn
        self.vector_store = vector_store_ref  # Keep reference for health checks

    def invoke(self, initial_state: dict) -> dict:
        """Execute the pipeline: retrieve -> generate"""
        state = initial_state.copy()

        # Retrieve step
        retrieve_result = self.retrieve(state)
        state.update(retrieve_result)

        # Generate step
        generate_result = self.generate(state)
        state.update(generate_result)

        return state

    def get_documents_count(self) -> int:
        """Get the number of indexed documents"""
        if hasattr(self.vector_store, 'documents'):
            return len(self.vector_store.documents)
        return 0