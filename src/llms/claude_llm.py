import os

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

DEFAULT_MODEL = "claude-sonnet-4-5"
FALLBACK_MODEL = "claude-haiku-4-5"
FAST_MODEL = "claude-haiku-4-5"
KEY_ENV = "ANTHROPIC_API_KEY"

TEMPLATE = """You are an IPL History & Analytics Expert.
Context: {context}
Question: {question}

Expert Answer:"""


def default_model():
    return os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)


def fallback_model():
    return os.getenv("CLAUDE_FALLBACK_MODEL", FALLBACK_MODEL)


def fast_model():
    return os.getenv("CLAUDE_FAST_MODEL", FAST_MODEL)


def get_chat_model(model=None, api_key=None, temperature=0, max_output_tokens=2048, **kwargs):
    if api_key:
        kwargs["api_key"] = api_key
    return ChatAnthropic(model=model or default_model(), temperature=temperature,
                         max_tokens=max_output_tokens, max_retries=1, **kwargs)


def get_chain(retriever, model=None, temperature=0, **kwargs):
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    llm = get_chat_model(model=model, temperature=temperature, **kwargs)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
