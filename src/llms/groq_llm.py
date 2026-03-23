import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

TEMPLATE = """You are an IPL History & Analytics Expert.
Context: {context}
Question: {question}

Expert Answer:"""


def get_chat_model(model=None, temperature=0, **kwargs):
    if model is None:
        model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    return ChatGroq(model=model, temperature=temperature, **kwargs)


def get_chain(retriever, model=None, temperature=0, **kwargs):
    if model is None:
        model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    llm = get_chat_model(model=model, temperature=temperature, **kwargs)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
