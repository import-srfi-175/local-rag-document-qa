# local-rag-document-qa
Local PDF-based Document Q&amp;A using RAG with Ollama and open-source embeddings.

A document Q&A assistant that:
- accepts a PDF
- answers only from that PDF
- explicitly states "I cannot find the answer..." instead of hallucinating
- runs fully locally

Non-negotiables:
- local llm via ollama
- vectors embeddings
- local vector db
- simple web ui
- repo + demo + readme