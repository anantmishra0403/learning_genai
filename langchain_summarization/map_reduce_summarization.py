import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain


load_dotenv()

model_name = "llama3.1:8b"
llm = ChatOllama(model=model_name,temperature=1)

loader = PyPDFLoader("learning_genai/langchain_summarization/attention.pdf")
documents = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(documents)

chunk_template = """
Write a concise summary of the following document
Document: {text}
"""

chunk_prompt = PromptTemplate(
    input_variables=["text"],
    template=chunk_template
)

final_template = """
You are a helpful assistant that summarizes documents and explains them properly.
Please summarize the following document in 5 points:
{text}
"""

final_prompt = PromptTemplate(
    input_variables=["text"],
    template=final_template
)

chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=chunk_prompt, combine_prompt=final_prompt, verbose=True)
summary = chain.run(final_documents)
print(summary)