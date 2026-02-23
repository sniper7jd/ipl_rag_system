from .openai_llm import generate as openai_generate, get_chain as openai_get_chain

PROVIDERS = {
    "openai": openai_get_chain,
}

try:
    from .gemini_llm import get_chain as gemini_get_chain
    PROVIDERS["gemini"] = gemini_get_chain
except ImportError:
    pass

try:
    from .claude_llm import get_chain as claude_get_chain
    PROVIDERS["claude"] = claude_get_chain
except ImportError:
    pass

try:
    from .groq_llm import get_chain as groq_get_chain
    PROVIDERS["groq"] = groq_get_chain
except ImportError:
    pass


def get_llm_chain(provider: str, retriever, **kwargs):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}. Options: {list(PROVIDERS.keys())}")
    return PROVIDERS[provider](retriever=retriever, **kwargs)
