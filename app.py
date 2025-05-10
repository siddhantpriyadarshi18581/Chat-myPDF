import streamlit as st  # Import necessary libraries
import os
import sys
import io
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Initialize Session State to store the vector database
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Ensure dataset directory exists and check for available PDFs
dataset_dir = "rag-dataset"
if not os.path.exists(dataset_dir):
    st.error(f"Dataset directory '{dataset_dir}' not found!")
    st.stop()

pdfs = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".pdf")]

# Load and preprocess PDFs if available
if pdfs:
    docs = []
    for pdf in pdfs:
        loader = PyMuPDFLoader(pdf)
        docs.extend(loader.load())
    
    # Split documents into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Initialize embedding model and FAISS vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    sample_vector = embeddings.embed_query("sample text")
    index = faiss.IndexFlatL2(len(sample_vector))
    vector_store = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    vector_store.add_documents(documents=chunks)
    
    # Store vector store in session state
    st.session_state.vector_store = vector_store
    st.success("✅ PDFs processed and indexed!")
else:
    st.error("No PDF files found in the dataset directory!")
    st.stop()

# Streamlit UI setup for user interaction
st.title("💬 Research-GPT")
st.subheader("Ask Questions")
query = st.text_input("Type your question:")

# Process user query using RAG approach
if query and st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 100, 'lambda_mult': 1})
    model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")

    # Define prompt template for structured responses
    prompt = ChatPromptTemplate.from_template(
        """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
        
        Question: {question} 
        Context: {context} 
        Answer:
        """
    )

    # Function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Define retrieval-augmented generation (RAG) chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    # Generate and display response
    try:
        response = rag_chain.invoke(query)
        st.write("### Answer:")
        st.markdown(response)
    except Exception as e:
        st.error(f"Error generating response: {e}")
        
        
        
# # Set-ExecutionPolicy Unrestricted -Scope Process



# import streamlit as st
# import os
# import io
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_ollama import OllamaEmbeddings, ChatOllama
# import faiss
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.prompts import ChatPromptTemplate

# # Set up dataset directory
# dataset_dir = "rag-dataset"
# os.makedirs(dataset_dir, exist_ok=True)

# # Upload PDF files through web interface
# st.title("💬 Research-GPT")
# st.subheader("📄 Upload PDFs for Context")
# uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# # Save uploaded files to dataset directory
# if uploaded_files:
#     for uploaded_file in uploaded_files:
#         file_path = os.path.join(dataset_dir, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#     st.success("PDFs uploaded and saved!")

# # Initialize Session State to store the vector database
# if "vector_store" not in st.session_state:
#     st.session_state.vector_store = None

# # Load and preprocess all PDFs from dataset
# pdfs = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".pdf")]

# if pdfs:
#     docs = []
#     for pdf in pdfs:
#         loader = PyMuPDFLoader(pdf)
#         docs.extend(loader.load())
    
#     # Split documents into smaller chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     chunks = text_splitter.split_documents(docs)

#     # Initialize embedding model and FAISS vector store
#     embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
#     sample_vector = embeddings.embed_query("sample text")
#     index = faiss.IndexFlatL2(len(sample_vector))
#     vector_store = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
#     vector_store.add_documents(documents=chunks)

#     # Store vector store in session state
#     st.session_state.vector_store = vector_store
#     st.success("✅ PDFs processed and indexed for Q&A!")
# else:
#     st.warning("No PDF files found in the dataset directory.")

# # Question-answering interface
# st.subheader("💡 Ask Questions")
# query = st.text_input("Type your question:")

# if query and st.session_state.vector_store:
#     retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 100, 'lambda_mult': 1})
#     model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")

#     prompt = ChatPromptTemplate.from_template(
#         """
#         You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
#         If you don't know the answer, just say that you don't know.
#         Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
        
#         Question: {question} 
#         Context: {context} 
#         Answer:
#         """
#     )

#     def format_docs(docs):
#         return "\n\n".join([doc.page_content for doc in docs])
    
#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | model
#         | StrOutputParser()
#     )

#     try:
#         response = rag_chain.invoke(query)
#         st.write("### Answer:")
#         st.markdown(response)
#     except Exception as e:
#         st.error(f"Error generating response: {e}")
