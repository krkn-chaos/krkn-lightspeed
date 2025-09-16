"""
Embedding model configuration and utilities
"""

# Created by Claude Sonnet 4

from typing import Any, Dict

from langchain_huggingface import HuggingFaceEmbeddings

# Popular embedding models with their configurations
EMBEDDING_MODELS = {
    "qwen-small": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "description": "Fast, lightweight Chinese/English embedding model",
        "dimensions": 512,
        "size": "600MB",
    },
    "sentence-transformers-mini": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Fast, small multilingual model",
        "dimensions": 384,
        "size": "80MB",
    },
    "sentence-transformers-base": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "description": "High quality general-purpose model",
        "dimensions": 768,
        "size": "420MB",
    },
    "bge-small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "description": "BAAI BGE small English model",
        "dimensions": 384,
        "size": "130MB",
    },
    "bge-base": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "description": "BAAI BGE base English model",
        "dimensions": 768,
        "size": "440MB",
    },
    "instructor": {
        "model_name": "hkunlp/instructor-base",
        "description": "Instruction-based embedding model",
        "dimensions": 768,
        "size": "440MB",
    },
}

# Chunking strategies for different document types
CHUNKING_STRATEGIES = {
    "default": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "description": "Balanced approach for general documents",
    },
    "small_chunks": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "description": "Smaller chunks for precise retrieval",
    },
    "large_chunks": {
        "chunk_size": 2000,
        "chunk_overlap": 400,
        "description": "Larger chunks for more context",
    },
    "code_docs": {
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "description": "Optimized for code documentation",
    },
    "academic_papers": {
        "chunk_size": 1200,
        "chunk_overlap": 240,
        "description": "Optimized for academic content",
    },
}


def get_embedding_model(
    model_key: str = "qwen-small", **kwargs
) -> HuggingFaceEmbeddings:
    """
    Get a configured embedding model

    Args:
        model_key: Key from EMBEDDING_MODELS or custom model name
        **kwargs: Additional arguments for HuggingFaceEmbeddings

    Returns:
        Configured HuggingFaceEmbeddings instance
    """
    if model_key in EMBEDDING_MODELS:
        model_name = EMBEDDING_MODELS[model_key]["model_name"]
        print(f"Using embedding model: {model_key} ({model_name})")
        print(f"Description: {EMBEDDING_MODELS[model_key]['description']}")
    else:
        model_name = model_key
        print(f"Using custom embedding model: {model_name}")

    return HuggingFaceEmbeddings(model_name=model_name, **kwargs)


def get_chunking_config(strategy_key: str = "default") -> Dict[str, Any]:
    """
    Get chunking configuration

    Args:
        strategy_key: Key from CHUNKING_STRATEGIES

    Returns:
        Dictionary with chunk_size and chunk_overlap
    """
    if strategy_key in CHUNKING_STRATEGIES:
        config = CHUNKING_STRATEGIES[strategy_key].copy()
        print(f"Using chunking strategy: {strategy_key}")
        print(f"Description: {config['description']}")
        config.pop("description")  # Remove description from config
        return config
    else:
        print(f"Unknown chunking strategy '{strategy_key}', using default")
        return CHUNKING_STRATEGIES["default"].copy()


def list_embedding_models():
    """List all available embedding models"""
    print("Available Embedding Models:")
    print("=" * 50)
    for key, config in EMBEDDING_MODELS.items():
        print(f"Key: {key}")
        print(f"  Model: {config['model_name']}")
        print(f"  Description: {config['description']}")
        print(f"  Dimensions: {config['dimensions']}")
        print(f"  Size: {config['size']}")
        print()


def list_chunking_strategies():
    """List all available chunking strategies"""
    print("Available Chunking Strategies:")
    print("=" * 50)
    for key, config in CHUNKING_STRATEGIES.items():
        print(f"Strategy: {key}")
        print(f"  Chunk Size: {config['chunk_size']}")
        print(f"  Overlap: {config['chunk_overlap']}")
        print(f"  Description: {config['description']}")
        print()
