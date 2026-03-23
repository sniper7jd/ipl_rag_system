from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

TEMPLATE = """You are an IPL History & Analytics Expert.
Context: {context}
Question: {question}

Expert Answer:"""


def get_chat_model(model="gpt-4o", temperature=0, **kwargs):
    return ChatOpenAI(model=model, temperature=temperature, **kwargs)


def get_chain(retriever, model="gpt-4o", temperature=0, **kwargs):
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    llm = get_chat_model(model=model, temperature=temperature, **kwargs)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def generate(retriever, question, **kwargs):
    chain = get_chain(retriever, **kwargs)
    return chain.invoke(question)
