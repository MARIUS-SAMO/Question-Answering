from config import MODEL_CHECKPOINT, EMBEDDING_MODEL_CHECKPOINT
from transformers import pipeline
import fitz
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_reader_pipeline():
    qa_pipeline = pipeline(
        task="question-answering",
        model=MODEL_CHECKPOINT,
        tokenizer=MODEL_CHECKPOINT
    )

    return qa_pipeline


def get_answer(qa_pipeline, docs_and_scores, query):

    list_answer = []
    for element in docs_and_scores:
        print(element)
        input_qa = {"question": query, "context": element[0].page_content}
        answer = qa_pipeline(input_qa)
        print(element[0].metadata)
        answer.update(element[0].metadata)
        answer.update({"context": element[0].page_content})
        list_answer.append(answer)
        print(answer)

    sorted_data = sorted(list_answer, key=lambda x: x['score'], reverse=True)

    return sorted_data


def get_retrieve_docs_and_scores(db, query):
    docs_and_scores = db.similarity_search_with_score(query)
    return docs_and_scores


def extract_text(path):
    doc = fitz.open(stream=path, filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text("text", flags=64)

    doc.close()

    return text


def extract_text_from_pdf(path):

    text = ""
    for pdf in path:
        text += extract_text(pdf.read())

    return text


def get_vectorstore(text_chunks):
    print(EMBEDDING_MODEL_CHECKPOINT)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_CHECKPOINT)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
