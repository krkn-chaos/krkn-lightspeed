import os
import json
import logging
import time
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store that mimics langchain VectorStore interface"""

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
        self.index = None
        self.documents = []
        self.last_search_metrics = {}
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
        with open(docs_path, "r") as f:
            self.documents = json.load(f)

        logger.info(f"Index loaded: {len(self.documents)} documents indexed")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents and return as langchain Documents"""
        if not self.index:
            raise RuntimeError("Index not loaded")

        # Time embedding creation
        embedding_start = time.time()
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        embedding_time = time.time() - embedding_start

        # Time FAISS search
        search_start = time.time()
        scores, indices = self.index.search(query_embedding, k)
        search_time = time.time() - search_start

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
                        "relevance_score": float(score),
                    },
                )
                documents.append(doc)

        # Store metrics for external access
        self.last_search_metrics = {
            "embedding_time": embedding_time,
            "search_time": search_time,
            "total_retrieval_time": embedding_time + search_time,
            "documents_found": len(documents),
            "avg_relevance_score": sum(doc.metadata.get('relevance_score', 0) for doc in documents) / len(documents) if documents else 0,
            "top_score": documents[0].metadata.get('relevance_score', 0) if documents else 0
        }


        return documents


class SimpleStateGraph:
    def __init__(self, retrieve_fn, generate_fn, vector_store_ref):
        self.retrieve = retrieve_fn
        self.generate = generate_fn
        self.vector_store = (
            vector_store_ref  # Keep reference for health checks
        )

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
        if hasattr(self.vector_store, "documents"):
            return len(self.vector_store.documents)
        return 0

    def get_last_search_metrics(self) -> Dict[str, Any]:
        """Get metrics from the last search operation"""
        if hasattr(self.vector_store, "last_search_metrics"):
            return self.vector_store.last_search_metrics
        return {}
