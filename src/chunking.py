# convert large str -> overlapping chunks that preserve context

def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start = end - chunk_overlap

        if start < 0:
            start = 0

    return chunks

if __name__ == "__main__":
    from ingest import load_pdf_text

    text = load_pdf_text("data/Employee-Handbook.pdf")
    chunks = split_text_into_chunks(text)

    print(f"Total chunks: {len(chunks)}")
    print("-" * 50)
    print(chunks[0])

