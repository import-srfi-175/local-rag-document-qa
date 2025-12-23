# local-rag-document-qa

Local PDF-based Document Q&A using RAG with Ollama and open-source embeddings.

A document Q&A assistant that:

* accepts a PDF document
* answers questions strictly using the uploaded PDF
* explicitly returns *"I cannot find the answer to that question in the provided document."* when the answer is not present
* runs fully locally for data privacy

## Requirements

* python 3.9+
* ollama installed and running locally

## Tech Stack

* LLM: Phi-3 via Ollama
* Embeddings: Sentence Transformers (`all-MiniLM-L6-v2`)
* Vector Database: ChromaDB (persistent local storage)
* Orchestration: LangChain
* UI: Streamlit

## Setup

Pull the required Ollama model:

```bash
ollama pull phi3
```

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Start the Ollama server:

```bash
ollama serve
```

## Run the Application

```bash
streamlit run app.py
```

Upload a PDF and ask questions through the web interface.

## Architecture Notes

* The document is split into 500-character chunks with 100-character overlap to balance context preservation and retrieval precision.
* Embeddings are generated once per document and stored locally to avoid recomputation.
* Only the top 3 most relevant chunks are passed to the LLM to keep inference fast and reduce noise.

## Project Structure

```text
.
├── app.py
├── src/
│   ├── ingest.py
│   ├── chunking.py
│   ├── vectorstore.py
│   ├── retriever.py
│   └── rag.py
├── requirements.txt
└── README.md
```

## Notes

* Uploaded PDFs and local vector data are not committed to the repository.
* The system does not use any external or paid APIs.

---
