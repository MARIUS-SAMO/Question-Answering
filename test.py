from langchain.document_loaders import PyMuPDFLoader, PyPDFLoader


def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_loader = PyMuPDFLoader(pdf)
        pdf_pages = pdf_loader.load()
        for page in pdf_pages:
            text += page.page_content

    return text
