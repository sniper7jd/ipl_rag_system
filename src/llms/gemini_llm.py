import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_MODEL = "gemini-3.5-flash"
# Free-tier quotas are per model, so each fallback adds its own daily allowance.
FALLBACK_MODEL = "gemini-2.5-flash,gemini-3.1-flash-lite"
FAST_MODEL = "gemini-2.5-flash"
KEY_ENV = "GOOGLE_API_KEY"

TEMPLATE = """You are an IPL History & Analytics Expert.
Context: {context}
Question: {question}

Expert Answer:"""


def default_model():
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def fallback_model():
    return os.getenv("GEMINI_FALLBACK_MODEL", FALLBACK_MODEL)


def fast_model():
    return os.getenv("GEMINI_FAST_MODEL", FAST_MODEL)


def get_chat_model(model=None, api_key=None, temperature=0, max_output_tokens=2048, **kwargs):
    model = model or default_model()
    if model.startswith("gemini-3"):
        # Gemini 3 is tuned for its default temperature; low values can cause looping.
        kwargs.setdefault("thinking_config", {"thinking_level": "low"})
        temperature = None
    elif model.startswith("gemini-2.5"):
        kwargs.setdefault("thinking_budget", 0)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if api_key:
        kwargs["google_api_key"] = api_key
    return ChatGoogleGenerativeAI(model=model, max_output_tokens=max_output_tokens, max_retries=1, **kwargs)


def get_chain(retriever, model=None, temperature=0, **kwargs):
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    llm = get_chat_model(model=model, temperature=temperature, **kwargs)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
