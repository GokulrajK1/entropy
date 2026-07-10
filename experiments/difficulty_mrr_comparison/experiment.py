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


class DifficultyMRRComparisonExperiment(Experiment):
    def __init__(
        self,
        rag: EntropyRAG,
        number_of_passages_retrieved_per_query: int = 10,
        query_ids=None,
    ):
        init_params = {k: v for k, v in locals().items() if k not in ("self", "rag")}
        self.rag = rag
        self.k = number_of_passages_retrieved_per_query
        self._start_run(__file__, init_params)
        dataset_path = Path(self.rag.dataset.data_path)
        self.sections = [
            {
                "label": "Straightforward",
                "alpha": 0.3,
                "query_file": dataset_path / "queries_straightforward.tsv",
            },
            {
                "label": "Moderate",
                "alpha": 0.5,
                "query_file": dataset_path / "queries_moderate.tsv",
            },
            {
                "label": "Complicated",
                "alpha": 0.7,
                "query_file": dataset_path / "queries_complicated.tsv",
            },
        ]

    def run(self):
        section_labels = []
        similarity_mrr_values = []
        utility_mrr_values = []
        section_results = []

        for section in self.sections:
            query_ids = self.load_query_ids_from_file(section["query_file"])
            query_ids = self.resolve_query_ids(self.rag.dataset, query_ids=query_ids)
            alpha = section["alpha"]

            similarity_mrrs = []
            utility_mrrs = []

            for query_id in query_ids:
                query = self.rag.dataset.queries[query_id]
                retrieved_passages = self.rag.retrieve_top_k_passages(query, k=self.k)

                if not retrieved_passages:
                    continue

                similarity_sorted = sorted(
                    retrieved_passages,
                    key=lambda passage: passage.similarity_score if passage.similarity_score is not None else 0.0,
                    reverse=True,
                )
                utility_sorted = sorted(
                    retrieved_passages,
                    key=lambda passage: UtilityFunction.U2(
                        passage.similarity_score if passage.similarity_score is not None else 0.0,
                        passage.entropy if passage.entropy is not None else 0.0,
                        alpha,
                    ),
                    reverse=True,
                )

                similarity_metric = self.rag.evaluate_relevant_passages(
                    [passage.id for passage in similarity_sorted],
                    query,
                    cutoffs=None,
                )
                utility_metric = self.rag.evaluate_relevant_passages(
                    [passage.id for passage in utility_sorted],
                    query,
                    cutoffs=None,
                )

                if similarity_metric is not None:
                    similarity_mrrs.append(similarity_metric.mrr)
                if utility_metric is not None:
                    utility_mrrs.append(utility_metric.mrr)

            avg_similarity_mrr = float(np.mean(similarity_mrrs)) if similarity_mrrs else 0.0
            avg_utility_mrr = float(np.mean(utility_mrrs)) if utility_mrrs else 0.0

            section_labels.append(f"{section['label']}\nα={alpha}")
            similarity_mrr_values.append(avg_similarity_mrr)
            utility_mrr_values.append(avg_utility_mrr)
            section_results.append({
                "label": section["label"],
                "alpha": alpha,
                "similarity_mrr": avg_similarity_mrr,
                "utility_mrr": avg_utility_mrr,
            })

            print(
                f"{section['label']}: similarity MRR={avg_similarity_mrr:.4f}, "
                f"utility MRR={avg_utility_mrr:.4f}, alpha={alpha}"
            )

        x_positions = np.arange(len(section_labels))
        bar_width = 0.35

        plt.figure(figsize=(10, 6))
        similarity_bars = plt.bar(
            x_positions - bar_width / 2,
            similarity_mrr_values,
            bar_width,
            label="Similarity",
            color="steelblue",
        )
        utility_bars = plt.bar(
            x_positions + bar_width / 2,
            utility_mrr_values,
            bar_width,
            label="Utility (U2)",
            color="darkorange",
        )

        plt.xlabel("Query Difficulty Section")
        plt.ylabel("MRR")
        plt.title(f"MRR Comparison by Query Difficulty (k={self.k})")
        plt.xticks(x_positions, section_labels)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        for bars in [similarity_bars, utility_bars]:
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )

        output_file = self.output_path / f"mrr_by_difficulty_{self.k}passages.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_file}")

        self.save_results({
            "sections": section_results,
            "plot_file": str(output_file),
        })

