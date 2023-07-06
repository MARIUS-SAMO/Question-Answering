import streamlit as st
from PyPDF2 import PdfReader
from markdown import markdown
from annotated_text import annotation
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceInstructEmbeddings
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain


from backend_utils import get_retrieve_docs_and_scores, extract_text_from_pdf, get_vectorstore, get_reader_pipeline, get_answer, get_text_chunks


def main():
    # from backend_utils import get_retrieve_docs_and_scores, extract_text_from_pdf
    st.set_page_config(page_title="Chat with your Documents",
                       page_icon=":books:")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    st.header("Chat with your Documents :books:")

    pdf_docs = st.file_uploader(
        "Upload your Documents", accept_multiple_files=True
    )
    if st.button("Process"):
        with st.spinner("Processing"):
            print(pdf_docs)
            raw_text = extract_text_from_pdf(pdf_docs)
            # st.write(raw_text)
            text_chunks = get_text_chunks(raw_text)
            st.session_state.vectorstore = get_vectorstore(text_chunks)

    user_question = st.text_input("Ask a question about your documents:")
    if st.button("Run") and user_question:
        print("here")
        with st.spinner("🧠 &nbsp;&nbsp; Performing neural search on documents..."):
            docs_and_scores = get_retrieve_docs_and_scores(
                db=st.session_state.vectorstore, query=user_question)

            qa_pipeline = get_reader_pipeline()
            predicted_answers = get_answer(
                qa_pipeline=qa_pipeline, docs_and_scores=docs_and_scores, query=user_question)

        # Display the results
        for result in predicted_answers:
            answer, context = result["answer"], result["context"]
            start_idx, end_idx = result["start"], result["end"]

            st.write(markdown("...")+context[:start_idx]+str(annotation(answer, "ANSWER", "#3e1c21", "white")) +
                     context[end_idx:], unsafe_allow_html=True)
            st.markdown(
                f"**Score:** {result['score']:.2f}"
            )

            # st.write(predicted_answers)


if __name__ == "__main__":
    main()
