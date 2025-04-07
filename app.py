import streamlit as st
import os
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

st.title("10-K Financial Report Analyzer")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4", openai_api_key=api_key)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.sidebar.title("Upload 10-K Reports")
    pdf_files = st.sidebar.file_uploader(
        "Upload your 10-K PDF files", type=['pdf'], accept_multiple_files=True)

    if pdf_files:
        with st.spinner("Processing your 10-K reports..."):
            # Combine all PDF texts
            combined_text = ""
            for pdf_file in pdf_files:
                combined_text += extract_text_from_pdf(pdf_file)
            
            # Get text chunks
            text_chunks = get_text_chunks(combined_text)
            
            # Create vector store
            vectorstore = get_vectorstore(text_chunks)
            
            # Create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)
            
            st.success("10-K reports processed successfully!")

    # Display chat interface
    st.write("Ask questions about the 10-K reports")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_question = st.chat_input("Ask a question about the 10-K reports...")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        
        if st.session_state.conversation:
            with st.spinner("Analyzing..."):
                response = st.session_state.conversation({'question': user_question})
                answer = response['answer']
                
                with st.chat_message("assistant"):
                    st.write(answer)
                
                # Update chat history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            st.warning("Please upload 10-K PDF files first.")

if __name__ == "__main__":
    main()

