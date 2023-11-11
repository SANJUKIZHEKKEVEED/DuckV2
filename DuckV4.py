#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
print(os.getcwd())
os.chdir(r"c:\Users\Sanju\LangChain")


# In[3]:


import os
import streamlit as st
import langchain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF library for working with PDFs

# Create a Streamlit app
st.title("🦣Doc BOT")

# Relative paths to the 'docs' and 'data' folders in the GitHub repository
DOCS_DIRECTORY = "files"
PERSIST_DIRECTORY = "vchroma"

# Create a place to input the OpenAI API Key
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Document loading and processing
documents = []
document_names = []

if openai_api_key:
    # Set the OpenAI API key if it has been provided
    os.environ['OPENAI_API_KEY'] = openai_api_key

    for file in os.listdir(DOCS_DIRECTORY):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DOCS_DIRECTORY, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            document_names.append(os.path.splitext(file)[0])
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = os.path.join(DOCS_DIRECTORY, file)
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
            document_names.append(os.path.splitext(file)[0])
        elif file.endswith('.txt'):
            text_path = os.path.join(DOCS_DIRECTORY, file)
            loader = TextLoader(text_path)
            documents.extend(loader.load())
            document_names.append(os.path.splitext(file)[0])

    # Create a RecursiveCharacterTextSplitter object
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=10, separators=["\n\n", "\n", "(?<=\. )", " ", ""])
    # Split the list of documents
    documents = text_splitter.split_documents(documents)
    embedding = OpenAIEmbeddings()
    # Create the vector database
    vectordb = Chroma.from_documents(documents, embedding=embedding)
    # Create the chat model
    pdf_qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.8, model_name="gpt-3.5-turbo"),
        vectordb.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True
    )

    chat_history = []
    st.write("Document BOT: Work smarter, not harder, with your documents 🪄")

    # Allow the user to select which document they want to query
    document_selection = st.selectbox("Select the document to query:", document_names)

    # Allow the user to input a prompt
    query = st.text_input("Query Corner:")

    # Add a submit button with a spinner
    if st.button("Submit"):
        with st.spinner("Answering..."):
            if query:
                result = pdf_qa({"question": query, "chat_history": chat_history, "context_documents": [document_selection]})
                st.write("Answer:", result["answer"])
                chat_history.append((query, result["answer"]))


# In[ ]:




