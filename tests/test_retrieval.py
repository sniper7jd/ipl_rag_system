from langchain_core.documents import Document

from src.agents.rag_agent import cited_numbers, linkify
from src.knowledge.base import KnowledgeBase
from src.retrieval.hybrid import SearchSettings, rrf_fuse, search
from src.retrieval.rerank import apply_token_budget

VOCAB = ["captain", "coach", "runs", "wickets", "squad", "title", "final", "stadium"]


class BagOfWordsEmbeddings:
    """Deterministic stand-in for the ONNX model so tests run offline."""

    def _vec(self, text):
        low = text.lower()
        return [float(low.count(w)) + 0.01 for w in VOCAB]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _kb():
    docs = [
        Document(page_content="Source: CSK 2026 > Overview\nCaptain: Ruturaj Gaikwad. Coach: Stephen Fleming.",
                 metadata={"title": "CSK 2026", "section": "Overview", "teams": ["Chennai Super Kings"], "seasons": [2026]}),
        Document(page_content="Source: MI 2026 > Overview\nCaptain: Hardik Pandya. Most runs: Ryan Rickelton.",
                 metadata={"title": "MI 2026", "section": "Overview", "teams": ["Mumbai Indians"], "seasons": [2026]}),
        Document(page_content="Source: CSK > History\nCSK won the title in 2010 and the final in 2011.",
                 metadata={"title": "CSK", "section": "History", "teams": ["Chennai Super Kings"], "seasons": [2010, 2011]}),
    ]
    return KnowledgeBase.from_documents(docs, BagOfWordsEmbeddings())


def test_rrf_rewards_agreement_between_rankers():
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60)
    assert [i for i, _ in fused][:2] == [1, 3]
    assert abs(fused[0][1] - (1 / 61 + 1 / 62)) < 1e-9


def test_metadata_filters_restrict_candidates():
    kb = _kb()
    assert kb.allowed_ids(["Chennai Super Kings"]) == {0, 2}
    assert kb.allowed_ids(["Chennai Super Kings"], 2026) == {0}
    assert kb.allowed_ids() is None


def test_hybrid_search_with_filter_and_fallback():
    kb = _kb()
    settings = SearchSettings(top_k=2, use_reranker=False, min_filtered=1)
    hits, trace = search(kb, "who was the captain", ["Mumbai Indians"], 2026, settings)
    assert trace["filter"]["applied"] and [h.chunk_index for h in hits] == [1]

    strict = SearchSettings(top_k=2, use_reranker=False, min_filtered=5)
    hits, trace = search(kb, "who was the captain", ["Mumbai Indians"], 2026, strict)
    assert not trace["filter"]["applied"] and "fallback" in trace["filter"]
    assert len(hits) == 2


def test_token_budget_keeps_at_least_one_chunk():
    docs = [Document(page_content="x" * 4000, metadata={"tokens": 1000}) for _ in range(3)]
    kept, used = apply_token_budget([(d, None) for d in docs], max_tokens=1500)
    assert len(kept) == 1 and used == 1000


def test_citations_are_parsed_and_linked():
    answer = "RCB won [1][3] and GT lost [2, 3]."
    assert cited_numbers(answer) == [1, 3, 2]
    sources = [{"n": 1, "url": "https://a"}, {"n": 2, "url": "https://b"}, {"n": 3, "url": None}]
    assert linkify(answer, sources) == "RCB won [[1]](https://a)[3] and GT lost [[2]](https://b)[3]."
