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
from experiment_helpers import normalize_values, normalized_entropy_lookup


class RAGBenchMRREvaluationExperiment(Experiment):
    """
    Evaluate MRR metric for RAGBench datasets (e.g., CovidQA) with specific alpha values.
    RAGBench datasets typically have 3 relevant passages per query.
    This experiment evaluates performance with alpha=0 (entropy-only) and alpha=0.2.
    """
    
    def __init__(
        self,
        rag: EntropyRAG,
        alphas: list = None,
        utility_function=None,
        alpha: float = None,
        alpha_values: str = None,
        number_of_queries: int = None,
        number_of_passages_retrieved_per_query: int = 3,
        query_ids: list = None,
    ):
        init_params = {k: v for k, v in locals().items() if k not in ("self", "rag")}
        self.rag = rag
        # Accept either `alphas` list directly or CLI-friendly `alpha`/`alpha_values` inputs
        if alphas is not None:
            self.alphas = alphas
        elif alpha_values is not None:
            parsed = [v.strip() for v in str(alpha_values).split(",") if v.strip()]
            self.alphas = [float(v) for v in parsed]
        elif alpha is not None:
            self.alphas = [float(alpha)]
        else:
            self.alphas = [0.0, 0.2]
        self.utility_function = utility_function if utility_function is not None else UtilityFunction.U2
        self.normalized_entropy = (
            normalized_entropy_lookup(self.rag) if self.utility_function is UtilityFunction.U3 else None
        )
        self.number_of_queries = number_of_queries
        self.number_of_passages_retrieved_per_query = number_of_passages_retrieved_per_query
        self.query_ids = query_ids
        self._start_run(__file__, init_params)

    def run(self):
        """Run MRR evaluation for specified alphas."""
        
        selected_query_ids = self.resolve_query_ids(
            self.rag.dataset,
            query_ids=self.query_ids,
            number_of_queries=self.number_of_queries,
        )
        
        print(f"\n{'='*70}")
        print(f"RAGBench MRR Evaluation")
        print(f"Dataset: {self.rag.dataset.__class__.__name__}")
        print(f"Number of queries: {len(selected_query_ids)}")
        print(f"Number of passages per query: {self.number_of_passages_retrieved_per_query}")
        print(f"Alpha values: {self.alphas}")
        print(f"{'='*70}\n")
        
        # Store results
        results = {
            'alpha': [],
            'mean_mrr': [],
            'std_mrr': [],
            'median_mrr': [],
            'min_mrr': [],
            'max_mrr': [],
            'queries_with_relevant': [],
            'queries_evaluated': [],
        }
        
        # Evaluate for each alpha
        for alpha in sorted(self.alphas):
            mrr_scores = []
            queries_with_relevant = 0
            queries_evaluated = 0
            
            print(f"Evaluating alpha={alpha:.2f}...")
            
            for qid, query_id in enumerate(selected_query_ids, start=1):
                if query_id not in self.rag.dataset.queries:
                    continue
                
                query = self.rag.dataset.queries[query_id]
                
                # Skip if no relevant passages for this query
                if query_id not in self.rag.dataset.answers:
                    continue
                
                relevant_passages = self.rag.dataset.answers[query_id].relevant_passage_ids
                if not relevant_passages:
                    continue
                
                queries_with_relevant += 1
                
                # Retrieve passages
                retrieved_passages = self.rag.retrieve_top_k_passages(
                    query,
                    k=self.number_of_passages_retrieved_per_query
                )
                
                if not retrieved_passages:
                    mrr_scores.append(0.0)
                    queries_evaluated += 1
                    continue
                
                # Sort by utility function
                if self.normalized_entropy is not None:
                    entropy_for = lambda p: self.normalized_entropy.get(p.id, 0.0)
                elif self.utility_function is UtilityFunction.U4:
                    per_query_normalized = dict(zip(
                        (p.id for p in retrieved_passages),
                        normalize_values(p.entropy if p.entropy is not None else 0.0 for p in retrieved_passages),
                    ))
                    entropy_for = lambda p: per_query_normalized.get(p.id, 0.0)
                else:
                    entropy_for = lambda p: p.entropy if p.entropy is not None else 0.0

                utility_sorted = sorted(
                    retrieved_passages,
                    key=lambda p: self.utility_function(
                        p.similarity_score if p.similarity_score is not None else 0.0,
                        entropy_for(p),
                        alpha,
                    ),
                    reverse=True
                )
                
                # Evaluate MRR
                metric = self.rag.evaluate_relevant_passages(
                    [p.id for p in utility_sorted],
                    query,
                    cutoffs=None
                )
                
                if metric is not None:
                    mrr_scores.append(metric.mrr)
                else:
                    mrr_scores.append(0.0)
                
                queries_evaluated += 1
                
                if qid % 50 == 0 or qid == len(selected_query_ids):
                    print(f"  Progress: {qid}/{len(selected_query_ids)} queries")
            
            # Store statistics
            if mrr_scores:
                results['alpha'].append(alpha)
                results['mean_mrr'].append(np.mean(mrr_scores))
                results['std_mrr'].append(np.std(mrr_scores))
                results['median_mrr'].append(np.median(mrr_scores))
                results['min_mrr'].append(np.min(mrr_scores))
                results['max_mrr'].append(np.max(mrr_scores))
                results['queries_with_relevant'].append(queries_with_relevant)
                results['queries_evaluated'].append(queries_evaluated)
                
                print(f"  Mean MRR: {results['mean_mrr'][-1]:.4f} ± {results['std_mrr'][-1]:.4f}")
                print(f"  Median MRR: {results['median_mrr'][-1]:.4f}")
                print(f"  Queries with relevant passages: {queries_with_relevant}")
                print()
        
        # Save results to CSV
        results_csv = self.output_path / "mrr_results.csv"
        with open(results_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'alpha', 'mean_mrr', 'std_mrr', 'median_mrr',
                    'min_mrr', 'max_mrr', 'queries_with_relevant', 'queries_evaluated'
                ]
            )
            writer.writeheader()
            
            for i in range(len(results['alpha'])):
                writer.writerow({
                    'alpha': f"{results['alpha'][i]:.2f}",
                    'mean_mrr': f"{results['mean_mrr'][i]:.6f}",
                    'std_mrr': f"{results['std_mrr'][i]:.6f}",
                    'median_mrr': f"{results['median_mrr'][i]:.6f}",
                    'min_mrr': f"{results['min_mrr'][i]:.6f}",
                    'max_mrr': f"{results['max_mrr'][i]:.6f}",
                    'queries_with_relevant': results['queries_with_relevant'][i],
                    'queries_evaluated': results['queries_evaluated'][i],
                })
        
        print(f"Results saved to: {results_csv}")
        results['results_csv'] = str(results_csv)

        # Create visualization
        if results['alpha']:
            plt.figure(figsize=(12, 6))
            
            # Plot mean MRR with error bars
            plt.errorbar(
                results['alpha'],
                results['mean_mrr'],
                yerr=results['std_mrr'],
                marker='o',
                linewidth=2,
                markersize=8,
                capsize=5,
                label='Mean MRR ± Std Dev',
                color='steelblue'
            )
            
            # Plot median for reference
            plt.plot(
                results['alpha'],
                results['median_mrr'],
                marker='s',
                linewidth=2,
                markersize=6,
                linestyle='--',
                label='Median MRR',
                color='orange'
            )
            
            plt.xlabel('Alpha', fontsize=12)
            plt.ylabel('MRR (Mean Reciprocal Rank)', fontsize=12)
            plt.title(f'RAGBench MRR vs Alpha\n{self.rag.dataset.__class__.__name__}', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1.0])
            
            plot_file = self.output_path / "mrr_vs_alpha.png"
            plt.savefig(str(plot_file), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Plot saved to: {plot_file}")
            results['plot_file'] = str(plot_file)

            # Print summary
            print(f"\n{'='*70}")
            print("MRR Evaluation Summary")
            print(f"{'='*70}")
            for i, alpha in enumerate(results['alpha']):
                improvement = None
                if i > 0:
                    improvement = (results['mean_mrr'][i] - results['mean_mrr'][0]) / results['mean_mrr'][0] * 100
                    print(f"Alpha {alpha:.2f}: Mean MRR = {results['mean_mrr'][i]:.4f} (Improvement: {improvement:+.2f}%)")
                else:
                    print(f"Alpha {alpha:.2f}: Mean MRR = {results['mean_mrr'][i]:.4f} (Baseline)")

        self.save_results(results)

