# this file accepts a pdf path -> reads all pages -> extracts text -> returns a large string
from pypdf import PdfReader

def load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)

    full_text = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text.append(text)

    return "\n".join(full_text)

if __name__ == "__main__":
    pdf_path = "data/Employee-Handbook.pdf"
    text = load_pdf_text(pdf_path)

    print("PDF loaded successfully")
    print("-" * 50)
    print(text[:500])  # print first 500 characters
