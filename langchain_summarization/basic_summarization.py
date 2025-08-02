import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

model_name = "llama3.1:8b"
llm = ChatOllama(model=model_name,temperature=1)

loader = PyPDFLoader("learning_genai/langchain_summarization/attention.pdf")
documents = loader.load()

#### Way 1 to summarize using LLM
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
chat_message = [
    SystemMessage(content="You are a helpful assistant that summarizes documents and explain it properly."),
    HumanMessage(content=f"Please summarize the following document and explain it in 5 points: \n Text: {documents}")
]

llm_response = llm(chat_message)
print(llm_response.content)

#### Way 2 to summarize using LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

generictemplate = """
Write a concise summary of the following document in 5 points:
{document}
How are context vectors are calculated in transformers?
Can you explain it in simple terms?
Do you have any pytorch code to demonstrate it?
"""

prompt = PromptTemplate(
    input_variables=["document"],
    template=generictemplate
)
chain = LLMChain(llm=llm, prompt=prompt)
summary = chain.run({"document": documents})
print(summary)

