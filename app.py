import os
import streamlit as st
import sqlite3
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Load environment variables
load_dotenv()

def add_vertical_space(space: int):
    for _ in range(space):
        st.write("\n")

with st.sidebar:
    st.title("LLM Chat App")
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    '''
    )
    st.write('Made with by [Prompt Engineer](https://youtube.com/@engineerprompt)')

def create_vector_store_table(conn):
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS vector_store
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  store_name TEXT,
                  vector BLOB)''')
    conn.commit()

def create_vector_store(chunks, store_name):
    try:
        conn = sqlite3.connect('my_database.db')
        create_vector_store_table(conn)
        c = conn.cursor()
        
        # Serialize and insert each vector into the database
        for idx, vec in enumerate(chunks):
            c.execute('INSERT INTO vector_store (store_name, vector) VALUES (?, ?)', (store_name, sqlite3.Binary(vec.tobytes())))
        
        conn.commit()
        conn.close()
        
    except sqlite3.Error as e:
        st.error(f"SQLite error while creating vector store: {e}")

def load_vector_store(store_name):
    try:
        conn = sqlite3.connect('my_database.db')
        create_vector_store_table(conn)
        c = conn.cursor()
        
        # Check if the table exists
        c.execute('''SELECT count(name) FROM sqlite_master WHERE type='table' AND name='vector_store' ''')
        if c.fetchone()[0] != 1:
            st.error("Table 'vector_store' does not exist.")
            return []
        
        # Retrieve vectors from the database
        c.execute('SELECT vector FROM vector_store WHERE store_name=?', (store_name,))
        vectors = [np.frombuffer(row[0], dtype=np.float32) for row in c.fetchall()]
        
        # Assume we know chunk size and overlap for reconstruction
        chunk_size = 1000
        chunk_overlap = 200
        chunks = []
        for i in range(0, len(vectors), chunk_size - chunk_overlap):
            chunks.append(vectors[i:i + chunk_size])
        
        conn.close()
        
        return chunks
    
    except sqlite3.Error as e:
        st.error(f"SQLite error while loading vector store: {e}")
        return []

def main():
    st.header("Chat with PDF")
  
    pdf = st.file_uploader("Upload the PDF", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len)
        
        chunks = text_splitter.split_text(text=text)
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            st.write("Embeddings Loaded from the Disk")
            chunks = load_vector_store(store_name)
        else:
            st.write("Embeddings Computation Completed")
            create_vector_store(chunks, store_name)
        
        query = st.text_input("Ask question about your Judicial PDF files: ")
        if query:
            docs = chunks  # Assuming chunks as input documents for similarity search
            llm = OpenAI(temperature=0)  # Adjust parameters as per your needs
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                st.write(response)

if __name__ == '__main__':
    main()
