from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class RetrievalMetrics:
    hit_at_k: float
    recall_at_k: float
    mrr: float


def _to_set(relevant_docs: Iterable[str]) -> set:
    return set(relevant_docs)


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """
    Hit@k: 1.0 if at least one relevant document is present
    in the top-k retrieved list, 0.0 otherwise.
    """
    rel = _to_set(relevant)
    if not rel:
        return 0.0
    return 1.0 if any(doc_id in rel for doc_id in retrieved) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """
    Recall@k: fraction of relevant documents that appear in the top-k list.
    """
    rel = _to_set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for doc_id in rel if doc_id in retrieved)
    return hits / len(rel)


def mrr_at_k(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """
    Mean Reciprocal Rank (single query variant):
    1 / rank of the first relevant document in the top-k list,
    or 0.0 if there is no relevant document.
    """
    rel = _to_set(relevant)
    if not rel:
        return 0.0
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel:
            return 1.0 / idx
    return 0.0


def evaluate_single_query(
    retrieved: Sequence[str],
    relevant_docs: Iterable[str],
) -> RetrievalMetrics:
    """
    Compute all retrieval metrics for a single query.
    """
    h = hit_at_k(retrieved, relevant_docs)
    r = recall_at_k(retrieved, relevant_docs)
    m = mrr_at_k(retrieved, relevant_docs)
    return RetrievalMetrics(hit_at_k=h, recall_at_k=r, mrr=m)
