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


class QueryEntropyDifficultySplitExperiment(Experiment):
    def __init__(
        self,
        rag: EntropyRAG,
        number_of_queries: int = 5000,
        random_seed: int = 42,
        query_ids=None,
    ):
        init_params = {k: v for k, v in locals().items() if k not in ("self", "rag")}
        self.rag = rag
        self.number_of_queries = number_of_queries
        self.random_seed = random_seed
        self.query_ids = query_ids
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

        histogram_path = self.output_path / "entropy_distribution_histogram.png"
        entropy_values = [row["entropy"] for row in ranked_rows]

        plt.figure(figsize=(10, 6))
        plt.hist(entropy_values, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
        plt.xlabel("Query Entropy")
        plt.ylabel("Count")
        plt.title(f"Query Entropy Distribution (n={total_queries})")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(histogram_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved entropy histogram: {histogram_path}")

        self.save_results({
            "straightforward_count": len(straightforward_rows),
            "moderate_count": len(moderate_rows),
            "complicated_count": len(complicated_rows),
            "straightforward_csv": str(straightforward_csv),
            "moderate_csv": str(moderate_csv),
            "complicated_csv": str(complicated_csv),
            "histogram_file": str(histogram_path),
        })

