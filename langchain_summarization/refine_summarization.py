###This summarization is same as the map_reduce_summarization but using rolling window among the documents.
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

chain = load_summarize_chain(llm, chain_type="refine", verbose=True)
summary = chain.run(final_documents)
print(summary)
# Note: The refine summarization chain is designed to iteratively refine the summary based on the input documents.
# It may not produce a concise summary in the same way as map_reduce or stuff summarization chains.
# Instead, it builds a summary by refining it with each document in the input set.
# This is useful for cases where you want to progressively improve the summary as more context is provided.
# The output may vary based on the model and the documents provided.    