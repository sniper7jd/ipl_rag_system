import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-5-mini"
FALLBACK_MODEL = "gpt-4.1-mini"
FAST_MODEL = "gpt-4.1-mini"
KEY_ENV = "OPENAI_API_KEY"

TEMPLATE = """You are an IPL History & Analytics Expert.
Context: {context}
Question: {question}

Expert Answer:"""


def default_model():
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def fallback_model():
    return os.getenv("OPENAI_FALLBACK_MODEL", FALLBACK_MODEL)


def fast_model():
    return os.getenv("OPENAI_FAST_MODEL", FAST_MODEL)


def get_chat_model(model=None, api_key=None, temperature=0, max_output_tokens=2048, **kwargs):
    model = model or default_model()
    if api_key:
        kwargs["api_key"] = api_key
    if not model.startswith(("gpt-5", "o")):
        kwargs["temperature"] = temperature
    return ChatOpenAI(model=model, max_completion_tokens=max_output_tokens, max_retries=1, **kwargs)


def get_chain(retriever, model=None, temperature=0, **kwargs):
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    llm = get_chat_model(model=model, temperature=temperature, **kwargs)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def generate(retriever, question, **kwargs):
    return get_chain(retriever, **kwargs).invoke(question)
