from .openai_eff import get_embeddings as openai_embeddings
from .huggingface_eff import get_embeddings as huggingface_embeddings

PROVIDERS = {
    "openai": openai_embeddings,
    "huggingface": huggingface_embeddings,
}

try:
    from .cohere_eff import get_embeddings as cohere_embeddings
    PROVIDERS["cohere"] = cohere_embeddings
except ImportError:
    pass



def get_embeddings(provider: str):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown embeddings provider: {provider}. Options: {list(PROVIDERS.keys())}")
    return PROVIDERS[provider]()
