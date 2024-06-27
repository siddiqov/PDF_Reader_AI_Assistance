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
from langchain.docstore.document import Document
import hashlib

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

def create_tables(conn):
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS vector_store
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  store_name TEXT,
                  vector BLOB)''')
    c.execute('''CREATE TABLE IF NOT EXISTS document_metadata
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  document_name TEXT,
                  document_hash TEXT)''')
    conn.commit()

def compute_hash(file_content):
    """Compute MD5 hash of the file content."""
    return hashlib.md5(file_content).hexdigest()

def document_exists(conn, document_name, document_hash):
    """Check if a document with the given name and hash already exists in the database."""
    c = conn.cursor()
    c.execute('SELECT id FROM document_metadata WHERE document_name=? AND document_hash=?', (document_name, document_hash))
    return c.fetchone() is not None

def save_document_metadata(conn, document_name, document_hash):
    """Save the document metadata to the database."""
    c = conn.cursor()
    c.execute('INSERT INTO document_metadata (document_name, document_hash) VALUES (?, ?)', (document_name, document_hash))
    conn.commit()

def create_vector_store(chunks, store_name):
    try:
        conn = sqlite3.connect('my_database.db')
        create_tables(conn)
        c = conn.cursor()

        # Convert text chunks to embeddings
        embeddings = OpenAIEmbeddings()
        vectors = embeddings.embed_documents(chunks)
        
        # Serialize and insert each vector into the database
        for vec in vectors:
            vec_np = np.array(vec, dtype=np.float32)
            c.execute('INSERT INTO vector_store (store_name, vector) VALUES (?, ?)', (store_name, sqlite3.Binary(vec_np.tobytes())))
        
        conn.commit()
        conn.close()
        
    except sqlite3.Error as e:
        st.error(f"SQLite error while creating vector store: {e}")

def load_vector_store(store_name):
    try:
        conn = sqlite3.connect('my_database.db')
        create_tables(conn)
        c = conn.cursor()
        
        # Check if the table exists
        c.execute('''SELECT count(name) FROM sqlite_master WHERE type='table' AND name='vector_store' ''')
        if c.fetchone()[0] != 1:
            st.error("Table 'vector_store' does not exist.")
            return []
        
        # Retrieve vectors from the database
        c.execute('SELECT vector FROM vector_store WHERE store_name=?', (store_name,))
        vectors = [np.frombuffer(row[0], dtype=np.float32) for row in c.fetchall()]
        
        conn.close()
        
        return vectors
    
    except sqlite3.Error as e:
        st.error(f"SQLite error while loading vector store: {e}")
        return []

def summarize_text(text):
    """Summarize text to reduce token count."""
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm.run(f"Please summarize the following text: {text}")
    return response

def main():
    st.header("Chat with PDF")
  
    pdf_files = st.file_uploader("Upload the PDF files", type='pdf', accept_multiple_files=True)
    if pdf_files:
        all_chunks = []
        conn = sqlite3.connect('my_database.db')
        create_tables(conn)
        
        for pdf in pdf_files:
            pdf_reader = PdfReader(pdf)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                length_function=len)
            
            chunks = text_splitter.split_text(text=text)
            
            # Summarize large chunks to reduce token count
            summarized_chunks = [summarize_text(chunk) if len(chunk.split()) > 500 else chunk for chunk in chunks]
            store_name = pdf.name[:-4]
            file_content = pdf.getvalue()
            document_hash = compute_hash(file_content)
            
            if document_exists(conn, pdf.name, document_hash):
                st.write(f"Embeddings Loaded from the Disk for {pdf.name}")
                chunks = load_vector_store(store_name)
            else:
                st.write(f"Embeddings Computation Completed for {pdf.name}")
                create_vector_store(summarized_chunks, store_name)
                save_document_metadata(conn, pdf.name, document_hash)
            
            all_chunks.extend(summarized_chunks)
        
        conn.close()

        query = st.text_input("Ask question about your Judicial PDF files: ")
        if query:
            # Convert all chunks to Document objects
            docs = [Document(page_content=chunk) for chunk in all_chunks]
            
            llm = OpenAI(temperature=0)  # Adjust parameters as per your needs
            chain = load_qa_chain(llm=llm, chain_type="map_reduce")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                st.write(response)
                # Display cost and token usage
                st.write(f"Total Tokens: {cb.total_tokens}")
                st.write(f"Prompt Tokens: {cb.prompt_tokens}")
                st.write(f"Completion Tokens: {cb.completion_tokens}")
                st.write(f"Total Cost (USD): ${cb.total_cost:.5f}")

if __name__ == '__main__':
    main()
