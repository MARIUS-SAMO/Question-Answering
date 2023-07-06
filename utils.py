import fitz


def extract_text(path):
    doc = fitz.open(path)
    text = ""

    for page in doc:
        text += page.get_text("text", flags=64)

    doc.close()

    return text


def extract_text_from_pdf(path):

    text = ""

    for pdf in path:
        text += extract_text(pdf)

    return text
