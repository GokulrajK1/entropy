from rag import EntropyCalculator, EntropyRAG, AutoregressiveEntropyCalculator, embed_texts, count_tokens_tiktoken
from rag import UtilityFunction
from rag_datasets import RAGDataset, MSMACRODataset, RAGBenchCovidQADataset, NaturalQuestionsDataset, Query, Passage, Answer
import argparse
import sys
import json
import csv
import os
import inspect
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ..base import Experiment


class QueryEntropyDifficultyMRRExperiment(Experiment):
    def __init__(
        self,
        rag: EntropyRAG,
        number_of_queries: int = 500,
        number_of_passages_retrieved_per_query: int = 10,
        random_seed: int = 42,
        query_ids=None,
    ):
        init_params = {k: v for k, v in locals().items() if k not in ("self", "rag")}
        self.rag = rag
        self.number_of_queries = number_of_queries
        self.k = number_of_passages_retrieved_per_query
        self.random_seed = random_seed
        self.query_ids = query_ids
        self.alphas = [i / 10 for i in range(0, 11)]
        self._start_run(__file__, init_params)

    def _sample_query_ids(self):
        if self.query_ids is not None:
            return self.resolve_query_ids(self.rag.dataset, query_ids=self.query_ids)

        all_query_ids = list(self.rag.dataset.queries.keys())
        if not all_query_ids:
            return []

        sample_size = min(self.number_of_queries, len(all_query_ids))
        if sample_size < self.number_of_queries:
            print(
                f"Warning: requested {self.number_of_queries} queries but only {len(all_query_ids)} are available. "
                f"Using {sample_size}."
            )

        rng = random.Random(self.random_seed)
        return rng.sample(all_query_ids, sample_size)

    def _write_query_csv(self, rows, file_name):
        csv_path = self.output_path / file_name
        with open(csv_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(["query_id", "query_entropy", "query_text"])
            for row in rows:
                writer.writerow([
                    row["query_id"],
                    f"{row['entropy']:.6f}",
                    row["query_text"],
                ])
        return csv_path

    def run(self):
        selected_query_ids = self._sample_query_ids()
        if not selected_query_ids:
            print("No queries available for the experiment.")
            self.save_results({"status": "no_queries_available"})
            return

        query_rows = []
        for query_id in selected_query_ids:
            query = self.rag.dataset.queries[query_id]
            entropy_value = float(self.rag.query_entropy_store[query_id])
            query_rows.append(
                {
                    "query_id": query_id,
                    "query_text": query.text,
                    "entropy": entropy_value,
                }
            )

        ranked_rows = sorted(query_rows, key=lambda row: row["entropy"])
        total_queries = len(ranked_rows)
        lower_cut = int(total_queries * 0.25)
        upper_cut = int(total_queries * 0.75)

        straightforward_rows = ranked_rows[:lower_cut]
        moderate_rows = ranked_rows[lower_cut:upper_cut]
        complicated_rows = ranked_rows[upper_cut:]

        straightforward_csv = self._write_query_csv(straightforward_rows, "queries_straightforward.csv")
        moderate_csv = self._write_query_csv(moderate_rows, "queries_moderate.csv")
        complicated_csv = self._write_query_csv(complicated_rows, "queries_complicated.csv")

        print(f"Saved straightforward queries: {len(straightforward_rows)} -> {straightforward_csv}")
        print(f"Saved moderate queries: {len(moderate_rows)} -> {moderate_csv}")
        print(f"Saved complicated queries: {len(complicated_rows)} -> {complicated_csv}")

        complicated_query_ids = [row["query_id"] for row in complicated_rows]
        if not complicated_query_ids:
            print("No complicated queries found; skipping MRR-vs-alpha plot.")
            self.save_results({
                "straightforward_count": len(straightforward_rows),
                "moderate_count": len(moderate_rows),
                "complicated_count": len(complicated_rows),
                "straightforward_csv": str(straightforward_csv),
                "moderate_csv": str(moderate_csv),
                "complicated_csv": str(complicated_csv),
                "status": "no_complicated_queries",
            })
            return

        avg_mrr_by_alpha = []
        for alpha in self.alphas:
            mrr_values = []
            for query_id in complicated_query_ids:
                query = self.rag.dataset.queries[query_id]
                retrieved_passages = self.rag.retrieve_top_k_passages(query, k=self.k)

                if not retrieved_passages:
                    continue

                utility_sorted = sorted(
                    retrieved_passages,
                    key=lambda passage: UtilityFunction.U2(
                        passage.similarity_score if passage.similarity_score is not None else 0.0,
                        passage.entropy if passage.entropy is not None else 0.0,
                        alpha,
                    ),
                    reverse=True,
                )

                metric = self.rag.evaluate_relevant_passages(
                    [passage.id for passage in utility_sorted],
                    query,
                    cutoffs=None,
                )
                if metric is not None:
                    mrr_values.append(metric.mrr)

            avg_mrr = float(np.mean(mrr_values)) if mrr_values else 0.0
            avg_mrr_by_alpha.append(avg_mrr)
            print(f"Complicated queries: alpha={alpha:.1f}, avg MRR={avg_mrr:.4f}")

        best_index = int(np.argmax(avg_mrr_by_alpha))
        best_alpha = self.alphas[best_index]
        best_mrr = avg_mrr_by_alpha[best_index]

        mrr_csv_path = self.output_path / "complicated_mrr_vs_alpha.csv"
        with open(mrr_csv_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(["alpha", "avg_mrr"])
            for alpha, mrr_value in zip(self.alphas, avg_mrr_by_alpha):
                writer.writerow([f"{alpha:.1f}", f"{mrr_value:.6f}"])

        plt.figure(figsize=(10, 6))
        plt.plot(self.alphas, avg_mrr_by_alpha, marker="o", linewidth=2, color="steelblue")
        plt.scatter([best_alpha], [best_mrr], color="crimson", s=90, zorder=3, label="Best alpha")
        plt.annotate(
            f"best a={best_alpha:.1f}, MRR={best_mrr:.3f}",
            xy=(best_alpha, best_mrr),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=10,
            color="crimson",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="crimson", alpha=0.8),
        )

        plt.xlabel("Alpha")
        plt.ylabel("MRR")
        plt.title(f"Complicated (Top 25% Entropy) MRR vs Alpha using U2 (k={self.k})")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()

        output_file = self.output_path / f"complicated_mrr_vs_alpha_u2_{self.k}passages.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved MRR-vs-alpha CSV: {mrr_csv_path}")
        print(f"Saved MRR-vs-alpha plot: {output_file}")

        self.save_results({
            "straightforward_count": len(straightforward_rows),
            "moderate_count": len(moderate_rows),
            "complicated_count": len(complicated_rows),
            "straightforward_csv": str(straightforward_csv),
            "moderate_csv": str(moderate_csv),
            "complicated_csv": str(complicated_csv),
            "alphas": self.alphas,
            "avg_mrr_by_alpha": avg_mrr_by_alpha,
            "best_alpha": best_alpha,
            "best_mrr": best_mrr,
            "mrr_csv": str(mrr_csv_path),
            "plot_file": str(output_file),
        })

