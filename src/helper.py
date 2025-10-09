import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

# ----------------- PDF PROCESSING -----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    if not text:
        raise ValueError("No text found in the uploaded PDF(s).")
    return text


# ----------------- TEXT SPLITTING -----------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    if not chunks:
        raise ValueError("Text chunks could not be created. Check PDF content.")
    return chunks


# ----------------- VECTOR STORE -----------------
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


# ----------------- CONVERSATION CHAIN -----------------
def get_conversational_chain(vector_store):
    chat_model = pipeline(
        "text-generation",
        model="distilgpt2",   
        max_new_tokens=150,  
        truncation=True,
        pad_token_id=50256    
    )
    llm = HuggingFacePipeline(pipeline=chat_model)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


# ----------------- SUMMARIZATION FEATURE -----------------
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text[:2000], max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']


# ----------------- QUIZ GENERATOR FEATURE -----------------
def generate_quiz(text):
    quiz_prompt = (
        "Generate 5 multiple-choice quiz questions with 4 options each "
        "based on the following content:\n"
        f"{text[:2000]}"
    )
    quiz_model = pipeline("text-generation", model="distilgpt2", max_new_tokens=300, pad_token_id=50256)
    quiz = quiz_model(quiz_prompt, max_new_tokens=300, truncation=True)
    return quiz[0]['generated_text']
