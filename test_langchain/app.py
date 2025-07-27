from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

def load_pdf_files(directory):
    """
    Load all PDF files from the specified directory and return their content as a list of strings.
    """
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    documents = []
    
    for pdf_file in pdf_files[:5]:
        print(pdf_file)
        try:
            loader = PyPDFLoader(os.path.join(directory, pdf_file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def get_ollama_embeddings(model_name="llama3.2:1b"):
    return OllamaEmbeddings(model=model_name)

def store_in_chroma(chunks, embedding_function, persist_dir="./chroma_store"):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_dir
    )
    return vectorstore

def retrieve_from_chroma(query, embedding_function, persist_dir="./chroma_store", top_k=3):
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_function
    )
    results = vectorstore.similarity_search(query, k=top_k)
    return results


documents = load_pdf_files(r'learning_genai/test_langchain/test_pdf_source')
chunks  = split_documents(documents)
embeddings = get_ollama_embeddings()
vectorstore = store_in_chroma(chunks, embeddings)
query = "What is the main theme of the documents?"
results = retrieve_from_chroma(query, embeddings)

for i, res in enumerate(results):
    print(f"\n--- Result {i+1} ---\n{res.page_content}\n")