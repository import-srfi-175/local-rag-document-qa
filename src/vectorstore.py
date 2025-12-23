# initialize embeddings -> create/load the chroma db -> store chunks

from sentence_transformers import SentenceTransformer
import chromadb


class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db"):
        # Load embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Use PersistentClient (THIS is the key fix)
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        # Create or load collection
        self.collection = self.client.get_or_create_collection(
            name="document_chunks"
        )

    def add_chunks(self, chunks: list[str]):
        embeddings = self.embedding_model.encode(chunks).tolist()
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )

if __name__ == "__main__":
    from ingest import load_pdf_text
    from chunking import split_text_into_chunks

    text = load_pdf_text("data/Employee-Handbook.pdf")
    chunks = split_text_into_chunks(text)

    store = VectorStore()
    store.add_chunks(chunks)

    print(f"Stored {len(chunks)} chunks in ChromaDB")
