"""LLM provider registry. Providers whose integration package is missing are skipped."""
import importlib

_MODULES = {
    "gemini": "src.llms.gemini_llm",
    "openai": "src.llms.openai_llm",
    "groq": "src.llms.groq_llm",
    "claude": "src.llms.claude_llm",
}

PROVIDER_MODULES = {}
for _name, _path in _MODULES.items():
    try:
        PROVIDER_MODULES[_name] = importlib.import_module(_path)
    except ImportError:
        pass

PROVIDERS = {name: mod.get_chain for name, mod in PROVIDER_MODULES.items()}
CHAT_MODEL_PROVIDERS = {name: mod.get_chat_model for name, mod in PROVIDER_MODULES.items()}


def _module(provider: str):
    if provider not in PROVIDER_MODULES:
        raise ValueError(f"Unknown LLM provider: {provider}. Options: {list(PROVIDER_MODULES)}")
    return PROVIDER_MODULES[provider]


def key_env(provider: str) -> str:
    return _module(provider).KEY_ENV


def default_model(provider: str) -> str:
    return _module(provider).default_model()


def fallback_models(provider: str) -> list:
    """Ordered fallback model names (comma-separated in the *_FALLBACK_MODEL setting)."""
    return [m.strip() for m in _module(provider).fallback_model().split(",") if m.strip()]


def get_chat_model(provider: str, **kwargs):
    return _module(provider).get_chat_model(**kwargs)


def get_fast_model(provider: str, api_key=None):
    """Small, low-latency model for auxiliary calls (query rewriting, judging)."""
    mod = _module(provider)
    return mod.get_chat_model(model=mod.fast_model(), api_key=api_key, max_output_tokens=256)


def get_llm_chain(provider: str, retriever, **kwargs):
    return _module(provider).get_chain(retriever=retriever, **kwargs)
