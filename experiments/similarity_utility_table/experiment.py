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


class SimilarityUtilityTableExperiment(Experiment):
    def __init__(
        self,
        rag: EntropyRAG,
        utility_function=None,
        number_of_queries: int = 10,
        number_of_passages_retrieved_per_query: int = 10,
        query_ids = None,
    ):
        init_params = {k: v for k, v in locals().items() if k not in ("self", "rag")}
        self.rag = rag
        self.utility_function = utility_function if utility_function is not None else UtilityFunction.U1
        self.number_of_queries = number_of_queries
        self.k = number_of_passages_retrieved_per_query
        self.query_ids = query_ids
        self.alphas = [0.0, 0.5]
        self._start_run(__file__, init_params)

    def run(self):
        query_ids = self.resolve_query_ids(
            self.rag.dataset,
            query_ids=self.query_ids,
            number_of_queries=self.number_of_queries,
        )

        csv_files = []
        for qid in query_ids:
            query = self.rag.dataset.queries[qid]
            retrieved = self.rag.retrieve_top_k_passages(query, k=self.k)

            relevant_set = set()
            if qid in self.rag.dataset.answers:
                relevant_set = self.rag.dataset.answers[qid].relevant_passage_ids

            # Create header with columns for each alpha value
            header = ["query_id"]
            for alpha in self.alphas:
                header.extend([
                    f"a={alpha}_rank",
                    f"a={alpha}_passage_id",
                    f"a={alpha}_utility_score",
                    f"a={alpha}_relevant",
                ])

            csv_path = self.output_path / f"query_{qid}.csv"

            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

                    # Sort passages by utility score for each alpha
                    sorted_by_alpha = {}
                    for alpha in self.alphas:
                        sorted_by_alpha[alpha] = sorted(
                            retrieved,
                            key=lambda p: self.utility_function(
                                p.similarity_score if p.similarity_score is not None else 0.0,
                                p.entropy if p.entropy is not None else 0.0,
                                alpha
                            ),
                            reverse=True,
                        )

                    for rank in range(self.k):
                        row = [qid]

                        for alpha in self.alphas:
                            if rank < len(sorted_by_alpha[alpha]):
                                p = sorted_by_alpha[alpha][rank]
                                utility_score = self.utility_function(
                                    p.similarity_score if p.similarity_score is not None else 0.0,
                                    p.entropy if p.entropy is not None else 0.0,
                                    alpha
                                )
                                utility_score_str = f"{utility_score:.6f}"
                                row.extend([
                                    str(rank + 1),
                                    p.id,
                                    utility_score_str,
                                    "1" if p.id in relevant_set else "0",
                                ])
                            else:
                                row.extend(["", "", "", ""])

                        writer.writerow(row)

                csv_files.append(str(csv_path))
                print(f"Wrote CSV for query {qid} -> {csv_path}")

            except Exception as e:
                print(f"Failed to write CSV for query {qid}: {e}")

        self.save_results({
            "query_ids": [str(qid) for qid in query_ids],
            "csv_files": csv_files,
        })

