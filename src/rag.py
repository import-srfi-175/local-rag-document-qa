# retrieve relevant chunks -> phi-3 via ollama -> strict prompting -> "answer not found"

SYSTEM_PROMPT = """
You are a document question-answering assistant.

Answer the user's question using ONLY the information provided in the context below.
If the answer is not explicitly present in the context, respond exactly with:
"I cannot find the answer to that question in the provided document."

Do NOT use any external knowledge.
Be concise and factual.
"""

from langchain_ollama import OllamaLLM
from retriever import Retriever


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = OllamaLLM(model="phi3", temperature=0)

    def answer(self, question: str) -> str:
        # 1. Retrieve relevant chunks
        chunks = self.retriever.retrieve(question)

        # 2. If nothing retrieved, return fallback
        if not chunks:
            return "I cannot find the answer to that question in the provided document."

        # 3. Build context
        context = "\n\n".join(chunks)

        # 4. Build full prompt
        prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
"""

        # 5. Generate answer
        response = self.llm.invoke(prompt)

        return response.strip()


if __name__ == "__main__":
    rag = RAGPipeline()

    print("---- QUESTION IN DOCUMENT ----")
    print(rag.answer("What is the leave policy?"))

    print("\n---- QUESTION NOT IN DOCUMENT ----")
    print(rag.answer("What is the CEO's favorite color?"))
