"""Reusable helper functions shared across experiments."""

import math


def select_top_entropy_queries(rag, percent):
    """Return the query ids with the top `percent`% highest query entropy.

    Ranks every query in `rag.dataset.queries` that has an entry in
    `rag.query_entropy_store` by entropy (descending) and returns the
    top `percent`% of them (at least 1).
    """
    if not 0 < percent <= 100:
        raise ValueError(f"percent must be in (0, 100], got {percent}")

    scored_query_ids = [
        qid for qid in rag.dataset.queries if qid in rag.query_entropy_store
    ]
    scored_query_ids.sort(key=lambda qid: rag.query_entropy_store[qid], reverse=True)

    count = max(1, math.ceil(len(scored_query_ids) * percent / 100))
    return scored_query_ids[:count]


def normalize_values(values):
    """Min-max normalize an iterable of floats to [0, 1] (all zeros if constant)."""
    values = list(values)
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return [0.0 for _ in values]
    return [(v - lo) / span for v in values]


def normalized_entropy_lookup(rag):
    """Return {passage_id: entropy} min-max normalized to [0, 1] from rag.entropy_store."""
    pids = list(rag.entropy_store.keys())
    normalized = normalize_values(rag.entropy_store[pid] for pid in pids)
    return dict(zip(pids, normalized))
