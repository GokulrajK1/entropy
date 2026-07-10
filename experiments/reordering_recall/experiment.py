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


class ReorderingRecallExperiment(Experiment):
    def __init__(self, rag : EntropyRAG, number_of_queries : int = 1000, number_of_passages_retrieved_per_query : int = 10, query_ids = None):
        init_params = {k: v for k, v in locals().items() if k not in ("self", "rag")}
        self.rag = rag
        self.number_of_queries = number_of_queries
        self.number_of_passages_retrieved_per_query = number_of_passages_retrieved_per_query
        self.query_ids = query_ids
        self._start_run(__file__, init_params)

    def run(self):
        
        alphas = [i / 10 for i in range(0, 11)]
        cutoffs = [3, 5, 10]
        
        metrics_by_alpha = {}
        
        selected_query_ids = self.resolve_query_ids(
            self.rag.dataset,
            query_ids=self.query_ids,
            number_of_queries=self.number_of_queries,
        )

        for alpha in alphas:
            metrics_by_alpha[alpha] = {
                'hit_recall': [],
                'overall_recall': [],
                'precision': [],
                'f1': [],
                'mrr': [],
                'dcg': [],
            }
            for cutoff in cutoffs:
                metrics_by_alpha[alpha][f'recall_at_{cutoff}'] = []
            
            utility_function = lambda similarity, entropy: alpha * similarity + (1 - alpha) * entropy
            print(f"Processing alpha={alpha:.1f}...")
            
            for qid, query_id in enumerate(selected_query_ids, start=1):
                query = self.rag.dataset.queries[query_id]
                retrieved_passages = self.rag.retrieve_top_k_passages(query, k=self.number_of_passages_retrieved_per_query)
                
                if not retrieved_passages:
                    continue
                
                # Sort by utility function
                utility_sorted = sorted(retrieved_passages, key=lambda p: utility_function(p.similarity_score, p.entropy), reverse=True)
                passage_ids = [p.id for p in utility_sorted]
                
                # Evaluate on all passages (no cutoffs for base metrics)
                base_metric = self.rag.evaluate_relevant_passages(passage_ids, query, cutoffs=None)
                
                if base_metric is None:
                    continue
                
                # Collect base metrics
                metrics_by_alpha[alpha]['hit_recall'].append(base_metric.hit_recall)
                metrics_by_alpha[alpha]['overall_recall'].append(base_metric.overall_recall)
                metrics_by_alpha[alpha]['precision'].append(base_metric.precision)
                metrics_by_alpha[alpha]['f1'].append(base_metric.f1)
                metrics_by_alpha[alpha]['mrr'].append(base_metric.mrr)
                metrics_by_alpha[alpha]['dcg'].append(base_metric.dcg)
                
                # Evaluate at specific cutoffs for recall@k
                cutoff_metric = self.rag.evaluate_relevant_passages(passage_ids, query, cutoffs=cutoffs)
                if cutoff_metric is not None:
                    for cutoff in cutoffs:
                        recall_k = cutoff_metric.recall_by_k.get(cutoff, 0.0)
                        metrics_by_alpha[alpha][f'recall_at_{cutoff}'].append(recall_k)
                
                if qid % 100 == 0:
                    print(f"  Processed {qid}/{len(selected_query_ids)} queries")
        
        print("\nGenerating plots...")
        # Create plots: one per metric
        base_metrics = ['hit_recall', 'overall_recall', 'precision', 'f1', 'mrr', 'dcg']

        metrics_avg_by_alpha = {}
        plot_files = []

        # For each base metric, create a graph with single line (average across all cutoffs)
        for metric_name in base_metrics:
            print(f"Creating plot for {metric_name}...")
            plt.figure(figsize=(10, 6))

            values = []
            for alpha in alphas:
                # Average metric across all cutoffs for this alpha
                cutoff_values = []
                alpha_values = metrics_by_alpha[alpha][metric_name]
                avg_val = np.mean(alpha_values) if alpha_values else 0.0
                values.append(avg_val)

            metrics_avg_by_alpha[metric_name] = values

            plt.plot(alphas, values, marker='o', linewidth=2, markersize=8, color='steelblue')

            plt.xlabel('Alpha', fontsize=12)
            plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
            plt.title(f'{metric_name.replace("_", " ").title()} vs Alpha', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])

            output_file = self.output_path / f"{metric_name}.png"
            plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
            plt.close()
            plot_files.append(str(output_file))
            print(f"Saved {output_file}")

        # For recall@k metrics, create one graph per cutoff (single line)
        for cutoff in cutoffs:
            print(f"Creating plot for recall_at_{cutoff}...")
            plt.figure(figsize=(10, 6))

            values = []
            for alpha in alphas:
                alpha_values = metrics_by_alpha[alpha][f'recall_at_{cutoff}']
                avg_val = np.mean(alpha_values) if alpha_values else 0.0
                values.append(avg_val)

            metrics_avg_by_alpha[f'recall_at_{cutoff}'] = values

            plt.plot(alphas, values, marker='o', linewidth=2, markersize=8, color='forestgreen')

            plt.xlabel('Alpha', fontsize=12)
            plt.ylabel(f'Recall@{cutoff}', fontsize=12)
            plt.title(f'Recall@{cutoff} vs Alpha', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])

            output_file = self.output_path / f"recall_at_{cutoff}.png"
            plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
            plt.close()
            plot_files.append(str(output_file))
            print(f"Saved {output_file}")

        print("\nExperiment complete!")

        self.save_results({
            "alphas": alphas,
            "metrics_avg_by_alpha": metrics_avg_by_alpha,
            "plot_files": plot_files,
        })

