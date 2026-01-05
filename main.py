import argparse
import os

from utils.state_graph import run_question_loop

# Available models and their pipeline loaders
AVAILABLE_MODELS = {
    "llama3.1": "rag_pipelines.llama31_rag_pipeline",
    "llama3.2": "rag_pipelines.llama32_rag_pipeline",
    "llama2.7": "rag_pipelines.llama27_rag_pipeline",
    "granite": "rag_pipelines.granite_rag_pipeline",
}

DEFAULT_MODEL = "llama3.1"


def get_model_choice():
    """Get model choice from command line args or environment variable."""
    parser = argparse.ArgumentParser(description="Krkn RAG Assistant")
    models = ', '.join(AVAILABLE_MODELS.keys())
    help_text = (
        f"Model to use. Available: {models}. "
        f"Can also set via KRKN_MODEL env var. Default: {DEFAULT_MODEL}"
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(AVAILABLE_MODELS.keys()),
        default=None,
        help=help_text
    )
    args = parser.parse_args()

    # Priority: CLI arg > env var > default
    model = args.model or os.environ.get("KRKN_MODEL", DEFAULT_MODEL)

    if model not in AVAILABLE_MODELS:
        available = ', '.join(AVAILABLE_MODELS.keys())
        print(f"Error: Unknown model '{model}'. Available: {available}")
        exit(1)

    return model


def load_pipeline(model_name):
    """Dynamically load the pipeline for the specified model."""
    if model_name == "llama3.1":
        from rag_pipelines.llama31_rag_pipeline import (
            load_llama31_rag_pipeline,
        )
        return load_llama31_rag_pipeline
    elif model_name == "llama3.2":
        from rag_pipelines.llama32_rag_pipeline import (
            load_llama32_rag_pipeline,
        )
        return load_llama32_rag_pipeline
    elif model_name == "llama2.7":
        from rag_pipelines.llama27_rag_pipeline import (
            load_llama27_rag_pipeline,
        )
        return load_llama27_rag_pipeline
    elif model_name == "granite":
        from rag_pipelines.granite_rag_pipeline import (
            load_granite_rag_pipline,
        )
        return load_granite_rag_pipline


def main():
    model = get_model_choice()
    print(f"Using model: {model}")

    github_repo = "https://github.com/krkn-chaos/website"
    repo_path = "content/en/docs"

    # Load the appropriate pipeline
    load_pipeline_fn = load_pipeline(model)

    # granite pipeline has different signature
    if model == "granite":
        graph = load_pipeline_fn()
    else:
        graph = load_pipeline_fn(github_repo, repo_path)

    # Run the question loop
    run_question_loop(graph)


if __name__ == "__main__":
    main()
