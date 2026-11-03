from .recursive import process as recursive_process
from .structured import process as structured_process
from .parent_child import process as parent_child_process
from .paragraph import process as paragraph_process

STRATEGIES = {
    "structured": structured_process,
    "recursive": recursive_process,
    "parent_child": parent_child_process,
    "paragraph": paragraph_process,
}

try:
    from .semantic import process as semantic_process
    STRATEGIES["semantic"] = semantic_process
except ImportError:
    pass


def get_chunker(strategy: str):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Options: {list(STRATEGIES.keys())}")
    return STRATEGIES[strategy]
