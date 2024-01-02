import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemp import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text 

def get_chunks(raw_text):
    splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embed = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embed) 
    return vectorstore

def get_conversation(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain
    
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv() 
    st.set_page_config(page_title='SomeLLM', page_icon=':robot_face:', layout='wide')
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("SomeLLM :robot_face:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Docs")
        pdf_docs = st.file_uploader("Upload", type=['txt', 'pdf'], accept_multiple_files=True, key='upload')
        if st.button("Process"):
            with st.spinner("Processing..."):
                #get the raw text
                raw_text = get_pdf_text(pdf_docs)
                #get the chunks
                text_chunks = get_chunks(raw_text)
                #create vector store
                vectorstore = get_vectorstore(text_chunks)
                #conversation chain instance
                st.session_state. conversation = get_conversation(vectorstore)
    
    

if __name__ == '__main__':
    main()
