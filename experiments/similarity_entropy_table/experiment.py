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


class SimilarityEntropyTableExperiment(Experiment):
    def __init__(
        self,
        rag: EntropyRAG,
        number_of_queries: int = 10,
        number_of_passages_retrieved_per_query: int = 10,
        query_ids = None,
    ):
        init_params = {k: v for k, v in locals().items() if k not in ("self", "rag")}
        self.rag = rag
        self.number_of_queries = number_of_queries
        self.k = number_of_passages_retrieved_per_query
        self.query_ids = query_ids
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

            sim_sorted = sorted(
                retrieved,
                key=lambda p: p.similarity_score if p.similarity_score is not None else 0.0,
                reverse=True,
            )
            ent_sorted = sorted(
                retrieved,
                key=lambda p: p.entropy if p.entropy is not None else 0.0,
                reverse=True,
            )

            relevant_set = set()
            if qid in self.rag.dataset.answers:
                relevant_set = self.rag.dataset.answers[qid].relevant_passage_ids

            header = [
                "query_id",
                "sim_rank", "sim_passage_id", "sim_score", "sim_relevant",
                "entropy_rank", "ent_passage_id", "entropy_score", "ent_relevant",
            ]

            csv_path = self.output_path / f"query_{qid}.csv"

            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

                    for rank in range(self.k):
                        sim_row = ["", "", "", ""]
                        ent_row = ["", "", "", ""]

                        if rank < len(sim_sorted):
                            p = sim_sorted[rank]
                            sim_score = (
                                f"{p.similarity_score:.6f}"
                                if p.similarity_score is not None
                                else ""
                            )
                            sim_row = [
                                str(rank + 1),
                                p.id,
                                sim_score,
                                "1" if p.id in relevant_set else "0",
                            ]

                        if rank < len(ent_sorted):
                            p2 = ent_sorted[rank]
                            ent_score = (
                                f"{p2.entropy:.6f}"
                                if p2.entropy is not None
                                else ""
                            )
                            ent_row = [
                                str(rank + 1),
                                p2.id,
                                ent_score,
                                "1" if p2.id in relevant_set else "0",
                            ]

                        row = [qid] + sim_row + ent_row
                        writer.writerow(row)

                csv_files.append(str(csv_path))
                print(f"Wrote CSV for query {qid} -> {csv_path}")

            except Exception as e:
                print(f"Failed to write CSV for query {qid}: {e}")

        self.save_results({
            "query_ids": [str(qid) for qid in query_ids],
            "csv_files": csv_files,
        })

