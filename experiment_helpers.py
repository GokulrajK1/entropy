"""Reusable helper functions shared across experiments."""

import math

import numpy as np


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


def evaluate_mrr_by_alpha(rag, query_ids, utility_function, alphas, k):
    """Evaluate MRR at each alpha for a fixed query set, re-ranking retrieved
    passages by `utility_function(similarity, entropy, alpha)`.

    Mirrors the per-query ranking/evaluation loop in
    `RAGBenchMRREvaluationExperiment`, factored out so other experiments can
    reuse it without duplicating the utility/entropy-normalization branching
    (raw entropy for U1/U2, corpus-wide normalized for U3, per-query
    normalized for U4).
    """
    from rag import UtilityFunction

    normalized_entropy = normalized_entropy_lookup(rag) if utility_function is UtilityFunction.U3 else None

    results = {
        "alpha": [], "mean_mrr": [], "std_mrr": [], "median_mrr": [],
        "min_mrr": [], "max_mrr": [], "queries_with_relevant": [], "queries_evaluated": [],
    }

    for alpha in sorted(alphas):
        mrr_scores = []
        queries_with_relevant = 0
        queries_evaluated = 0

        for query_id in query_ids:
            if query_id not in rag.dataset.queries:
                continue
            query = rag.dataset.queries[query_id]
            if query_id not in rag.dataset.answers:
                continue
            relevant_passages = rag.dataset.answers[query_id].relevant_passage_ids
            if not relevant_passages:
                continue
            queries_with_relevant += 1

            retrieved_passages = rag.retrieve_top_k_passages(query, k=k)
            if not retrieved_passages:
                mrr_scores.append(0.0)
                queries_evaluated += 1
                continue

            if normalized_entropy is not None:
                entropy_for = lambda p: normalized_entropy.get(p.id, 0.0)
            elif utility_function is UtilityFunction.U4:
                per_query_normalized = dict(zip(
                    (p.id for p in retrieved_passages),
                    normalize_values(p.entropy if p.entropy is not None else 0.0 for p in retrieved_passages),
                ))
                entropy_for = lambda p: per_query_normalized.get(p.id, 0.0)
            else:
                entropy_for = lambda p: p.entropy if p.entropy is not None else 0.0

            utility_sorted = sorted(
                retrieved_passages,
                key=lambda p: utility_function(
                    p.similarity_score if p.similarity_score is not None else 0.0,
                    entropy_for(p),
                    alpha,
                ),
                reverse=True,
            )

            metric = rag.evaluate_relevant_passages([p.id for p in utility_sorted], query, cutoffs=None)
            mrr_scores.append(metric.mrr if metric is not None else 0.0)
            queries_evaluated += 1

        if mrr_scores:
            results["alpha"].append(alpha)
            results["mean_mrr"].append(float(np.mean(mrr_scores)))
            results["std_mrr"].append(float(np.std(mrr_scores)))
            results["median_mrr"].append(float(np.median(mrr_scores)))
            results["min_mrr"].append(float(np.min(mrr_scores)))
            results["max_mrr"].append(float(np.max(mrr_scores)))
            results["queries_with_relevant"].append(queries_with_relevant)
            results["queries_evaluated"].append(queries_evaluated)

    return results
