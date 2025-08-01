# krkn-lightspeed
This is a RAG chatbot built using LangGraph, LangChain, and either the IBM Granite model, LLaMA 3.1 via Ollama, or LLama 2.7. The chatbot answers technical questions based on the KRKN pod scenarios documentation.

Note: To ensure accurate responses based on the provided documentation, please include the keyword “krkn” or other krkn context in your questions. This helps the system retrieve relevant context from the Krkn knowledge base, rather than generating general answers from unrelated sources.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/tejugang/krkn-lightspeed-rag-chatbot.git
cd krkn-lightspeed-rag-chatbot
```

### 2. Create + activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
## Running the Chatbot

If using the llama 3.1 generative model, run this script: 
```bash
brew install ollama
ollama run llama3
```
Ensure that ollama is running in the background

If using the llama 2.7 model, create a folder in the project's base directory called "models", [download the model](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf) and add it to the "models" folder in the project

### Terminal Interface
1. open main.py and uncomment the code for the generative model you would like to use
2. run ```python3 main.py``` (depending on your python version)


### UI Interface
1. run ```streamlit run app.py ```

## Performance
LLM performance improves significantly with better laptop hardware. LLM was tested on two different laptops: 
1. **Laptop 1**: Apple M3 Pro, 36 GB RAM, 12-core CPU, 18-core GPU
2. **Laptop 2**: Apple M1, 16 GB RAM, 8-core CPU, 12-core GPU

Answers were generated in **under 10 seconds** on laptop 1, whereas answers were generated in **15-30 seconds** on laptop 2. (for llama 3.1 LLM)

## Roadmap
Enhancements being planned can be found in the [roadmap](roadmap.md)

## Evaluating the model
[User guide to the evaluation pipeline](https://docs.google.com/document/d/1Z8KLLzhMC8zJf-aQJg4LkeROuzAB71A5U3-HyiICo8g/edit?tab=t.0)

Note: The output of steps 1-3 are the files in the folder ```evaluationPipeline ```

1. open eval.py and uncomment the code for the model you are evaluating
2. edit the email field on line 121 with the email that evaluation metrics should be sent to
3. after the script runs, open the json file (file name is on line 125)
4. copy the entire json file and open the [Evaluation Pipeline Endpoint](https://evaluation-api-rhsc-ai.apps.int.spoke.preprod.us-east-1.aws.paas.redhat.com/docs#/) (must connected to VPN). 
5. make sure the json structure matches the required format in the endpoint and paste it in these three endpoints```/evaluate_context_retrieval```, ```evaluate_response```, and ```evaluate_all```
6. evaluation metrics should be emailed to you

[Evaluation data](https://drive.google.com/drive/folders/1pLRgeLMEEvxacZML3B7Ges5nnsJr4t-W?usp=drive_link)
