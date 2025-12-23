# embed the question -> query the chromadb -> retrieve the top k relevant chunks -> return as plain text
from sentence_transformers import SentenceTransformer
import chromadb

class Retriever:
    def __init__(self, persist_directory: str = "chroma_db"):
        # Load embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load persistent chromadb
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_collection(
            name="document_chunks"
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        # Retrieve top-k relevant document chunks for a query.
        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Chroma returns lists inside lists
        documents = results.get("documents", [[]])[0]

        return documents

if __name__ == "__main__":
    retriever = Retriever()

    question = "What is the leave policy?"
    chunks = retriever.retrieve(question)

    print(f"Retrieved {len(chunks)} chunks")
    print("-" * 50)

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:\n{chunk[:300]}\n")
