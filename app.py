import streamlit as st
from src.helper import (
    get_pdf_text, 
    get_text_chunks, 
    get_vector_store, 
    get_conversational_chain,
    summarize_text,
    generate_quiz
)
import time 

def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("🧑 User:", message.content)
        else:
            st.write("🤖 Bot:", message.content)

def main():
    st.set_page_config(page_title="AI Study Buddy")
    st.header("📚 AI-Powered Study Buddy\nInformation-Retrieval-System 🤖⚡💡")

    user_question = st.text_input("Ask a question from the PDF")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""

    if user_question and st.session_state.conversation:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF files", 
            accept_multiple_files=True
        )

        if st.button("🚀 Submit & Process"):
            with st.spinner("Processing your PDF..."):
                time.sleep(2)
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.session_state.raw_text = raw_text
                st.success("✅ Processing Complete!")

        # New Buttons for Added Features
        if st.session_state.raw_text:
            if st.button("🧾 Summarize PDF"):
                with st.spinner("Summarizing..."):
                    summary = summarize_text(st.session_state.raw_text)
                    st.subheader("📘 Summary:")
                    st.write(summary)

            if st.button("🧠 Generate Quiz"):
                with st.spinner("Generating Quiz..."):
                    quiz = generate_quiz(st.session_state.raw_text)
                    st.subheader("🎯 Quiz Questions:")
                    st.write(quiz)

if __name__ == "__main__":
    main()
