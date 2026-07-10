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


class EntropyThresholdTopPassageMRRExperiment(Experiment):
    def __init__(
        self,
        rag: EntropyRAG,
        entropy_csv_files: str,
        entropy_threshold: float = 10.0,
        utility_function=None,
        alpha: float = 0.2,
        alpha_values: str = None,
        number_of_passages_retrieved_per_query: int = 10,
    ):
        init_params = {k: v for k, v in locals().items() if k not in ("self", "rag")}
        self.rag = rag
        self.entropy_threshold = float(entropy_threshold)
        self.utility_function = utility_function if utility_function is not None else UtilityFunction.U2
        if alpha_values is not None:
            parsed_alphas = [value.strip() for value in str(alpha_values).split(",") if value.strip()]
            self.alphas = [float(value) for value in parsed_alphas]
        else:
            self.alphas = [float(alpha)]
        self.k = number_of_passages_retrieved_per_query
        self.csv_files = [Path(f.strip()) for f in entropy_csv_files.split(",")]
        self._start_run(__file__, init_params)

    def _load_queries_by_entropy_threshold(self):
        query_rows_by_id = {}

        for csv_path in self.csv_files:
            if not csv_path.exists():
                print(f"Warning: CSV file not found, skipping: {csv_path}")
                continue

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue

                required_columns = {"query_id", "query_entropy", "query_text"}
                if not required_columns.issubset(set(reader.fieldnames)):
                    print(
                        f"Warning: CSV {csv_path} must contain columns {sorted(required_columns)}. "
                        f"Found: {reader.fieldnames}. Skipping."
                    )
                    continue

                for row in reader:
                    qid = str(row.get("query_id", "")).strip()
                    entropy_str = str(row.get("query_entropy", "")).strip()
                    qtext = str(row.get("query_text", "")).strip()
                    if not qid or not entropy_str:
                        continue

                    try:
                        entropy_val = float(entropy_str)
                    except ValueError:
                        continue

                    if entropy_val >= self.entropy_threshold:
                        query_rows_by_id[qid] = {
                            "query_id": qid,
                            "query_entropy": entropy_val,
                            "query_text": qtext,
                        }

        filtered_query_ids = list(query_rows_by_id.keys())
        resolved_query_ids = self.resolve_query_ids(self.rag.dataset, query_ids=filtered_query_ids)
        resolved_set = set(str(qid) for qid in resolved_query_ids)

        resolved_rows = [
            row for qid, row in query_rows_by_id.items()
            if str(qid) in resolved_set
        ]
        resolved_rows.sort(key=lambda row: int(row["query_id"]))

        return resolved_rows

    @staticmethod
    def _alpha_suffix(alpha: float) -> str:
        alpha_text = f"{alpha:g}"
        return alpha_text.replace("-", "neg_").replace(".", "_")

    @staticmethod
    def _passage_suffix(k: int) -> str:
        return f"{int(k)}passages"

    def run(self):
        selected_queries = self._load_queries_by_entropy_threshold()

        if not selected_queries:
            print(f"No query ids found with entropy >= {self.entropy_threshold}.")
            self.save_results({
                "status": "no_queries_found",
                "entropy_threshold": self.entropy_threshold,
            })
            return

        passage_suffix = self._passage_suffix(self.k)
        if len(self.alphas) == 1:
            output_csv = self.output_path / (
                f"entropy_threshold_{self.entropy_threshold:.2f}_{passage_suffix}_alpha_{self.alphas[0]:.2f}_top_passage_mrr.csv"
            )
        else:
            output_csv = self.output_path / (
                f"entropy_threshold_{self.entropy_threshold:.2f}_{passage_suffix}_alphas_top_passage_mrr.csv"
            )

        with open(output_csv, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            header = ["query_id", "query_text"]
            for alpha in self.alphas:
                suffix = self._alpha_suffix(alpha)
                header.extend([
                    f"passage_id_{suffix}",
                    f"passage_text_{suffix}",
                    f"mrr_{suffix}",
                ])
            writer.writerow(header)

            for idx, row in enumerate(selected_queries, start=1):
                query_id = str(row["query_id"])
                query = self.rag.dataset.queries[query_id]
                retrieved_passages = self.rag.retrieve_top_k_passages(query, k=self.k)

                row_values = [query_id, query.text]

                for alpha in self.alphas:
                    if not retrieved_passages:
                        row_values.extend(["", "", f"{0.0:.6f}"])
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

                    top_passage = utility_sorted[0]
                    metric = self.rag.evaluate_relevant_passages(
                        [passage.id for passage in utility_sorted],
                        query,
                        cutoffs=None,
                    )
                    mrr_value = metric.mrr if metric is not None else 0.0

                    row_values.extend([
                        top_passage.id,
                        top_passage.text,
                        f"{mrr_value:.6f}",
                    ])

                writer.writerow(row_values)

                if idx % 100 == 0:
                    print(f"Processed {idx}/{len(selected_queries)} queries")

        print(
            f"Saved entropy-threshold top-passage MRR CSV for {len(self.alphas)} alpha value(s) and {self.k} passages/query: {output_csv}"
        )

        self.save_results({
            "entropy_threshold": self.entropy_threshold,
            "alphas": self.alphas,
            "query_count": len(selected_queries),
            "csv_file": str(output_csv),
        })

