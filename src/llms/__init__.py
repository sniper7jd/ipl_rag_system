from .openai_llm import generate as openai_generate, get_chain as openai_get_chain, get_chat_model as openai_get_chat_model

PROVIDERS = {"openai": openai_get_chain}
CHAT_MODEL_PROVIDERS = {"openai": openai_get_chat_model}

try:
    from .gemini_llm import get_chain as gemini_get_chain, get_chat_model as gemini_get_chat_model
    PROVIDERS["gemini"] = gemini_get_chain
    CHAT_MODEL_PROVIDERS["gemini"] = gemini_get_chat_model
except ImportError:
    pass

try:
    from .claude_llm import get_chain as claude_get_chain, get_chat_model as claude_get_chat_model
    PROVIDERS["claude"] = claude_get_chain
    CHAT_MODEL_PROVIDERS["claude"] = claude_get_chat_model
except ImportError:
    pass

try:
    from .groq_llm import get_chain as groq_get_chain, get_chat_model as groq_get_chat_model
    PROVIDERS["groq"] = groq_get_chain
    CHAT_MODEL_PROVIDERS["groq"] = groq_get_chat_model
except ImportError:
    pass


def get_chat_model(provider: str, **kwargs):
    if provider not in CHAT_MODEL_PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}. Options: {list(CHAT_MODEL_PROVIDERS.keys())}")
    return CHAT_MODEL_PROVIDERS[provider](**kwargs)


def get_llm_chain(provider: str, retriever, **kwargs):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}. Options: {list(PROVIDERS.keys())}")
    return PROVIDERS[provider](retriever=retriever, **kwargs)
