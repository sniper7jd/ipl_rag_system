from .recursive import process as recursive_process
from .semantic import process as semantic_process
from .parent_child import process as parent_child_process
from .paragraph import process as paragraph_process

STRATEGIES = {
    "recursive": recursive_process,
    "semantic": semantic_process,
    "parent_child": parent_child_process,
    "paragraph": paragraph_process,
}


def get_chunker(strategy: str):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Options: {list(STRATEGIES.keys())}")
    return STRATEGIES[strategy]
