import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain


load_dotenv()

model_name = "llama3.1:8b"
llm = ChatOllama(model=model_name,temperature=1)

loader = PyPDFLoader("learning_genai/langchain_summarization/attention.pdf")
documents = loader.load_and_split()

template = """
Write a concise summary of the following document
Document: {text}
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=template
)

chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, verbose=True)
summary = chain.run(documents)
print(summary)