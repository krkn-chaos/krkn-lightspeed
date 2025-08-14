from rag_pipelines.llama31_rag_pipeline import load_llama31_rag_pipeline
from utils.state_graph import run_question_loop

# UNCOMMENT THE CODE FOR THE MODEL THAT YOU ARE NOT USING BEFORE RUNNING

urls = [
    "https://krkn-chaos.dev/docs/",
    "https://krkn-chaos.dev/docs/krkn/",
    "https://krkn-chaos.dev/docs/krkn-hub/",
    "data",
]
# START OF LLAMA 3.1 MODEL LOGIC
# llama 3.1
graph = load_llama31_rag_pipeline(urls)

# run in a loop
run_question_loop(graph)

# END OF LLAMA 3.1 MODEL LOGIC


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
