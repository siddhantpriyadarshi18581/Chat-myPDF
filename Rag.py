import os
import warnings
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import tiktoken

from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
print(load_dotenv())

# loader = PyMuPDFLoader(r"C:\Users\siddh\Desktop\Chat_PDF\rag-dataset\RoadSense.pdf")
# docs = loader.load()
# doc = docs[0]
# print(doc.metadata), 
# print(doc.page_content), 

pdfs = []
for root, dirs, files in os.walk('rag-dataset'):
    for file in files:
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(root, file))
            
# print(pdfs)

docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    pages = loader.load()
    
    
    docs.extend(pages)
    
# print(len(docs))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

chunks = text_splitter.split_documents(docs)

# print(len(docs), len(chunks))

# print(chunks[2].page_content)

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

# print(len(encoding.encode(docs[0].page_content)), len(encoding.encode(chunks[0].page_content)))



embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

single_vector = embeddings.embed_query("this is the text data for research Gpt")


# print(single_vector)

index = faiss.IndexFlatL2(len(single_vector))
# print(index.ntotal, index.d)


vector_store = FAISS(embedding_function = embeddings,
                     index = index,
                     docstore = InMemoryDocstore(),
                     index_to_docstore_id={})

ids = vector_store.add_documents(documents=chunks)

# print(len(ids))

# question = "what are the dataset used in for training of model in CyberShield-AI?"
# docs = vector_store.search(query=question, search_type='similarity')

# for doc in docs:
#     print(doc.page_content)
#     print("\n\n")
    
    
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs = {'k': 3, 'fetch_k':100, 'lambda_mult': 1})

# # question = "What is the CyberShield-AI model doing?"
# # question = "What is Phishing"
# # question = "What is road Safety?"
# question = "What are the benefits of Lane Detection"
# docs = retriever.invoke(question)

# for doc in docs:
#     print(doc.page_content)
#     print("\n\n")

model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")

print(model.invoke("hi"))


prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
    Question: {question} 
    Context: {context} 
    Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# print(format_docs(docs))


rag_chain = (
    {"context": retriever|format_docs, "question": RunnablePassthrough()}
     | prompt
     | model
     | StrOutputParser()
)
