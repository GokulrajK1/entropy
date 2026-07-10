from rag import EntropyCalculator, EntropyRAG, AutoregressiveEntropyCalculator, embed_texts, count_tokens_tiktoken
from rag import UtilityFunction
from rag_datasets import RAGDataset, MSMACRODataset, RAGBenchCovidQADataset, NaturalQuestionsDataset, Query, Passage, Answer
import argparse
import sys
import json
import csv
import os
import inspect
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ..base import Experiment


class DifficultyRankingComplicatedAlphaSweepExperiment(Experiment):
    def __init__(
        self,
        rag: EntropyRAG,
        difficulty_ranking_file: str,
        utility_function=None,
        number_of_passages_retrieved_per_query: int = 10,
        difficulty_label: str = "complicated",
    ):
        init_params = {k: v for k, v in locals().items() if k not in ("self", "rag")}
        self.rag = rag
        self.difficulty_ranking_file = Path(self.rag.dataset.data_path) / Path(difficulty_ranking_file)
        self.utility_function = utility_function if utility_function is not None else UtilityFunction.U2
        self.k = number_of_passages_retrieved_per_query
        self.difficulty_label = difficulty_label.strip().lower()
        self.alphas = [i / 10 for i in range(0, 11)]
        self._start_run(__file__, init_params)

    def _load_query_ids_by_difficulty(self):
        query_ids = []
        if not self.difficulty_ranking_file.exists():
            raise FileNotFoundError(f"Difficulty ranking file not found: {self.difficulty_ranking_file}")

        with open(self.difficulty_ranking_file, "r", encoding="utf-8") as file_handle:
            reader = csv.DictReader(file_handle)
            if not reader.fieldnames:
                return query_ids

            required_columns = {"query_id", "difficulty"}
            if not required_columns.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"Difficulty ranking CSV must contain columns {sorted(required_columns)}. "
                    f"Found: {reader.fieldnames}"
                )

            for row in reader:
                qid = str(row.get("query_id", "")).strip()
                difficulty = str(row.get("difficulty", "")).strip().lower()
                if qid and difficulty == self.difficulty_label:
                    query_ids.append(qid)

        return query_ids

    def run(self):
        ranked_query_ids = self._load_query_ids_by_difficulty()
        ranked_query_ids = self.resolve_query_ids(self.rag.dataset, query_ids=ranked_query_ids)

        if not ranked_query_ids:
            print(
                f"No query ids found for difficulty '{self.difficulty_label}' "
                f"in {self.difficulty_ranking_file}."
            )
            self.save_results({
                "difficulty_label": self.difficulty_label,
                "query_count": 0,
                "status": "no_queries_found",
            })
            return

        print(
            f"Loaded {len(ranked_query_ids)} '{self.difficulty_label}' queries "
            f"from {self.difficulty_ranking_file}."
        )

        avg_mrr_by_alpha = []
        for alpha in self.alphas:
            mrr_values = []
            for idx, query_id in enumerate(ranked_query_ids, start=1):
                query = self.rag.dataset.queries[query_id]
                retrieved_passages = self.rag.retrieve_top_k_passages(query, k=self.k)

                if not retrieved_passages:
                    continue

                utility_sorted = sorted(
                    retrieved_passages,
                    key=lambda passage: self.utility_function(
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

                if idx % 100 == 0:
                    print(
                        f"  alpha={alpha:.1f}: processed {idx}/{len(ranked_query_ids)} "
                        f"{self.difficulty_label} queries"
                    )

            avg_mrr = float(np.mean(mrr_values)) if mrr_values else 0.0
            avg_mrr_by_alpha.append(avg_mrr)
            print(f"alpha={alpha:.1f}, avg MRR={avg_mrr:.4f}")

        best_index = int(np.argmax(avg_mrr_by_alpha))
        best_alpha = self.alphas[best_index]
        best_mrr = avg_mrr_by_alpha[best_index]

        output_stem = f"{self.difficulty_label}_mrr_vs_alpha"
        mrr_csv_path = self.output_path / f"{output_stem}.csv"
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
        plt.title(f"{self.difficulty_label.title()} MRR vs Alpha using Utility (k={self.k})")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()

        output_file = self.output_path / f"{output_stem}_{self.k}passages.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Best alpha={best_alpha:.1f}, best MRR={best_mrr:.4f}")
        print(f"Saved MRR-vs-alpha CSV: {mrr_csv_path}")
        print(f"Saved MRR-vs-alpha plot: {output_file}")

        self.save_results({
            "difficulty_label": self.difficulty_label,
            "query_count": len(ranked_query_ids),
            "avg_mrr_by_alpha": avg_mrr_by_alpha,
            "best_alpha": best_alpha,
            "best_mrr": best_mrr,
            "csv_file": str(mrr_csv_path),
            "plot_file": str(output_file),
        })

