# krkn-lightspeed
This is a RAG chatbot built using LangGraph, LangChain, and either the IBM Granite model, LLaMA 3.1 via Ollama, or LLama 2. The chatbot answers technical questions based on the KRKN pod scenarios documentation.

Note: To ensure accurate responses based on the provided documentation, please include the keyword “krkn” or other krkn context in your questions. This helps the system retrieve relevant context from the Krkn knowledge base, rather than generating general answers from unrelated sources.


# Setup Instructions

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

### On macOS: 
If using the llama 3.1 LLM (reccomended), run this script: 
```bash
brew install ollama
ollama run llama3
```

If using llama 2:7b LLM, run this script: 
```bash
brew install ollama
ollama pull llama2:7b
```
### Other operating systems :
Download instructions [here](https://ollama.com/download)

Ensure that ollama is running in the background

### Terminal Interface
1. open main.py and uncomment the code for the LLM you would like to use
2. run ```python3 main.py``` (depending on your python version)


### UI Interface
1. run ```streamlit run app.py ```

## Steps 
1. **Document Processing**: The system loads and processes documentation files from the `docs/` directory under github.com/krkn-chaos/website, splitting them into manageable chunks for efficient retrieval.
    - Documents can be loaded as: 
        1. PDF (stored in a specific folder)
        2. Markdown files
        3. Urls 

2. **Vector Database Creation**: Document chunks are converted into embeddings using HuggingFace's sentence transformers and stored in a Chroma vector database for semantic search.

3. **RAG Pipeline Setup**: A Retrieval-Augmented Generation (RAG) pipeline is established using LangGraph and LangChain, combining document retrieval with language model generation.

4. **Model Integration**: The chatbot integrates with your chosen LLM (IBM Granite, LLaMA 3.1 via Ollama, or LLaMA 2) to generate contextually relevant responses.

5. **Query Processing**: When you ask a question, the system:
   - Retrieves relevant document chunks from the vector database
   - Provides context to the language model
   - Generates an answer based on the retrieved KRKN documentation
   - Returns the response with source citations when available

6. **Interactive Chat**: The terminal interface allows for continuous conversation, maintaining context throughout the session.


## Roadmap
Enhancements being planned can be found in the [roadmap](roadmap.md)

# Evaluation and Performance


## Performance
LLM performance improves significantly with better laptop hardware. LLM was tested on two different laptops: 
1. **Laptop 1**: Apple M3 Pro, 36 GB RAM, 12-core CPU, 18-core GPU
2. **Laptop 2**: Apple M1, 16 GB RAM, 8-core CPU, 12-core GPU

Answers were generated in **under 10 seconds** on laptop 1, whereas answers were generated in **15-30 seconds** on laptop 2. (for llama 3.1 LLM)


## Evaluating the model
If you want to evaluate the performance of the LLM being used to generate answers: 
[User guide to the evaluation pipeline](https://docs.google.com/document/d/1Z8KLLzhMC8zJf-aQJg4LkeROuzAB71A5U3-HyiICo8g/edit?tab=t.0)

Note: The output of steps 1-3 are the files in the folder ```evaluationPipeline ```

1. open eval.py and uncomment the code for the model you are evaluating
2. edit the email field on line 121 with the email that evaluation metrics should be sent to
3. after the script runs, open the json file (file name is on line 125)
4. copy the entire json file and open the [Evaluation Pipeline Endpoint](https://evaluation-api-rhsc-ai.apps.int.spoke.preprod.us-east-1.aws.paas.redhat.com/docs#/) (must connected to VPN). 
5. make sure the json structure matches the required format in the endpoint and paste it in these three endpoints```/evaluate_context_retrieval```, ```evaluate_response```, and ```evaluate_all```
6. evaluation metrics should be emailed to you

[Evaluation data](https://drive.google.com/drive/folders/1pLRgeLMEEvxacZML3B7Ges5nnsJr4t-W?usp=drive_link)


