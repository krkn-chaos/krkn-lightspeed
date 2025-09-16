# Modified by Claude Sonnet 4
import argparse
from rag_pipelines.llama31_rag_pipeline import load_llama31_rag_pipeline

from utils.state_graph import run_question_loop

# Parse command line arguments
parser = argparse.ArgumentParser(description='KRKN Lightspeed RAG Chatbot')
parser.add_argument('--krknctl', action='store_true', 
                   help='Use krknctl mode with llama.cpp and pre-downloaded model')
args = parser.parse_args()

github_repo="https://github.com/krkn-chaos/website"
repo_path="content/en/docs"
if args.krknctl:
    print("Starting in krknctl mode with llama.cpp...")
    graph = load_llama31_rag_pipeline(llm_backend="llamacpp")
else:
    print("Starting in default mode with Ollama...")
    graph = load_llama31_rag_pipeline(llm_backend="ollama")
# run in a loop
run_question_loop(graph)


"""#START OF GRANITE MODEL LOGIC
#granite
graph = load_granite_rag_pipline()

# run in a loop
run_question_loop(graph)

#END OF GRANITE MODEL LOGIC


#START OF LLAMA 2.7 MODEL LOGIC
#llama 2.7
graph = load_llama27_rag_pipeline()

# run in a loop
run_question_loop(graph)

#END OF LLAMA 2.7 MODEL LOGIC
"""
