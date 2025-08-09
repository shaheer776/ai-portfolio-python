
import os
import streamlit as st
import pickle
import time
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title('News Research Tool 📈')
st.sidebar.title('News Article URLs')

SAVE_PATH = 'C:/Users/shahe/OneDrive/Documents/GitHub/ai_portfolio/Generative_AI'
urls = []
embeddings = OpenAIEmbeddings()     # Create embeddings instance
llm = OpenAI(temperature=0.9, max_tokens=500)

for i in range(3):
    url = st.sidebar.text_input(f'URL {i+1}')
    urls.append(url)

process_url_clicked = st.sidebar.button('Process URLs')

main_placeholder = st.empty()

if process_url_clicked:
    # Load Data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split Data
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", ","], chunk_size=1000)

    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create Embeddings
    vectorstore_openai = FAISS.from_documents(docs, embedding=embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Store FAISS vector index to .pkl file
    vectorstore_openai.save_local(SAVE_PATH)

query = main_placeholder.text_input('Question: ')
process_query_clicked = st.button('Process Query')

if process_query_clicked:
    if os.path.exists(SAVE_PATH):
        vectorstore = FAISS.load_local(SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({'question': query}, return_only_outputs=True)   # {'answer': '', 'sources': []}
        st.header('Answer')
        st.write(result['answer'])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)

