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


class DifficultyAlphaSweepExperiment(Experiment):
    def __init__(
        self,
        rag: EntropyRAG,
        utility_function=None,
        number_of_passages_retrieved_per_query: int = 10,
        query_ids=None,
    ):
        init_params = {k: v for k, v in locals().items() if k not in ("self", "rag")}
        self.rag = rag
        self.utility_function = utility_function if utility_function is not None else UtilityFunction.U2
        self.k = number_of_passages_retrieved_per_query
        self.query_ids = query_ids
        self._start_run(__file__, init_params)
        dataset_path = Path(self.rag.dataset.data_path)
        self.sections = [
            {
                "label": "Straightforward",
                "query_file": dataset_path / "queries_straightforward.tsv",
                "output_name": "straightforward",
            },
            {
                "label": "Moderate",
                "query_file": dataset_path / "queries_moderate.tsv",
                "output_name": "moderate",
            },
            {
                "label": "Complicated",
                "query_file": dataset_path / "queries_complicated.tsv",
                "output_name": "complicated",
            },
        ]
        self.alphas = [i / 10 for i in range(0, 11)]

    def run(self):
        section_results = []
        for section in self.sections:
            query_ids = self.load_query_ids_from_file(section["query_file"])
            query_ids = self.resolve_query_ids(self.rag.dataset, query_ids=query_ids)

            avg_mrr_by_alpha = []
            for alpha in self.alphas:
                mrr_values = []
                for query_id in query_ids:
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

                avg_mrr = float(np.mean(mrr_values)) if mrr_values else 0.0
                avg_mrr_by_alpha.append(avg_mrr)

            if avg_mrr_by_alpha:
                best_index = int(np.argmax(avg_mrr_by_alpha))
                best_alpha = self.alphas[best_index]
                best_mrr = avg_mrr_by_alpha[best_index]
            else:
                best_index = 0
                best_alpha = self.alphas[0]
                best_mrr = 0.0

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
            plt.title(f"{section['label']} MRR vs Alpha (k={self.k})")
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend()

            output_file = self.output_path / f"{section['output_name']}_mrr_vs_alpha_{self.k}passages.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()

            print(
                f"{section['label']}: best alpha={best_alpha:.1f}, best MRR={best_mrr:.4f}. "
                f"Saved {output_file}"
            )

            section_results.append({
                "label": section["label"],
                "best_alpha": best_alpha,
                "best_mrr": best_mrr,
                "avg_mrr_by_alpha": avg_mrr_by_alpha,
                "plot_file": str(output_file),
            })

        self.save_results({
            "alphas": self.alphas,
            "sections": section_results,
        })

