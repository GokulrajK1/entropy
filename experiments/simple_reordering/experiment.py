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


class SimpleReorderingExperiment(Experiment):
    def __init__(self, rag : EntropyRAG, number_of_queries : int = 50, number_of_passages_retrieved_per_query : int = 3, query_ids = None):
        init_params = {k: v for k, v in locals().items() if k not in ("self", "rag")}
        self.rag = rag
        self.number_of_queries = number_of_queries
        self.number_of_passages_retrieved_per_query = number_of_passages_retrieved_per_query
        self.query_ids = query_ids
        self._start_run(__file__, init_params)

    def run(self):
        alphas = [i / 10 for i in range(0, 11)]
        alpha_to_number_reordered = {}
        selected_query_ids = self.resolve_query_ids(
            self.rag.dataset,
            query_ids=self.query_ids,
            number_of_queries=self.number_of_queries,
        )

        for alpha in alphas:
            utility_function = lambda similarity, entropy: alpha * similarity + (1 - alpha) * entropy
            reordered_count = 0

            for query_id in selected_query_ids:
                query = self.rag.dataset.queries[query_id]
                retrieved_passages = self.rag.retrieve_top_k_passages(query, k=self.number_of_passages_retrieved_per_query)

                if not retrieved_passages:
                    continue

                sim_sorted = sorted(retrieved_passages, key=lambda p: p.similarity_score, reverse=True)
                util_sorted = sorted(retrieved_passages, key=lambda p: utility_function(p.similarity_score, p.entropy), reverse=True)

                sim_order = [p.id for p in sim_sorted]
                util_order = [p.id for p in util_sorted]

                if sim_order != util_order:
                    reordered_count += 1

            alpha_to_number_reordered[alpha] = reordered_count

        print("Number of reordered queries per alpha:", alpha_to_number_reordered)

        plt.figure()
        plt.plot(alphas, [alpha_to_number_reordered[a] for a in alphas], marker='o')
        plt.title("Number of Queries with Reordered Passages vs Alpha")
        plt.xlabel("Alpha")
        plt.ylabel("Number of Reordered Queries")
        plt.grid(True)
        output_file = self.output_path / "reorder_vs_alpha_test_all.png"
        plt.savefig(output_file)

        self.save_results({
            "alphas": alphas,
            "alpha_to_number_reordered": {f"{alpha:.1f}": count for alpha, count in alpha_to_number_reordered.items()},
            "plot_file": str(output_file),
        })

