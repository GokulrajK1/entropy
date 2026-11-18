import csv

import numpy as np
import matplotlib.pyplot as plt

from rag import EntropyRAG, UtilityFunction
from ..base import Experiment
from experiment_helpers import select_top_entropy_queries, evaluate_mrr_by_alpha


class EntropyPercentileMRRSweepExperiment(Experiment):
    """
    Splits queries into top-N%-highest-query-entropy tiers (nested/cumulative,
    e.g. top 25%/15%/5%), then sweeps alpha and evaluates MRR on each tier
    under a chosen utility function. Produces one CSV+plot per tier plus a
    combined comparison plot, so the effect of entropy-aware reranking can be
    compared across query difficulty tiers.
    """

    def __init__(
        self,
        rag: EntropyRAG,
        percentiles: str = "25,15,5",
        alpha_values: str = "0.0,0.5,1.0,1.5,2.0,3.0",
        utility_function=None,
        number_of_passages_retrieved_per_query: int = 10,
    ):
        init_params = {k: v for k, v in locals().items() if k not in ("self", "rag")}
        self.rag = rag
        self.percentiles = [float(v.strip()) for v in str(percentiles).split(",") if v.strip()]
        self.alphas = [float(v.strip()) for v in str(alpha_values).split(",") if v.strip()]
        self.utility_function = utility_function if utility_function is not None else UtilityFunction.U2
        self.number_of_passages_retrieved_per_query = number_of_passages_retrieved_per_query
        self._start_run(__file__, init_params)

    def run(self):
        tier_results = {}
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(self.percentiles)))

        for pct, color in zip(self.percentiles, colors):
            query_ids = select_top_entropy_queries(self.rag, pct)
            print(f"\n{'='*70}")
            print(f"Entropy percentile tier: top {pct:g}% ({len(query_ids)} queries)")
            print(f"{'='*70}")

            results = evaluate_mrr_by_alpha(
                self.rag, query_ids, self.utility_function, self.alphas,
                self.number_of_passages_retrieved_per_query,
            )

            tier_label = f"top{pct:g}pct"
            tier_results[tier_label] = dict(results, percent=pct, num_queries=len(query_ids))

            csv_path = self.output_path / f"mrr_results_{tier_label}.csv"
            fieldnames = [
                "alpha", "mean_mrr", "std_mrr", "median_mrr",
                "min_mrr", "max_mrr", "queries_with_relevant", "queries_evaluated",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(results["alpha"])):
                    writer.writerow({name: results[name][i] for name in fieldnames})
            tier_results[tier_label]["csv_file"] = str(csv_path)

            plt.figure(figsize=(10, 6))
            plt.errorbar(
                results["alpha"], results["mean_mrr"], yerr=results["std_mrr"],
                marker="o", linewidth=2, markersize=7, capsize=4, label="Mean MRR ± Std",
            )
            plt.plot(results["alpha"], results["median_mrr"], marker="s", linestyle="--", label="Median MRR")
            plt.xlabel("Alpha")
            plt.ylabel("MRR")
            plt.title(
                f"MRR vs Alpha — top {pct:g}% highest-entropy queries ({len(query_ids)} queries)\n"
                f"{self.rag.dataset.__class__.__name__}"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1.0])
            tier_plot_path = self.output_path / f"mrr_vs_alpha_{tier_label}.png"
            plt.savefig(str(tier_plot_path), dpi=150, bbox_inches="tight")
            plt.close()
            tier_results[tier_label]["plot_file"] = str(tier_plot_path)

            if results["mean_mrr"]:
                print(f"  Mean MRR range: {min(results['mean_mrr']):.4f} - {max(results['mean_mrr']):.4f}")

        plt.figure(figsize=(12, 6))
        for (tier_label, r), color in zip(tier_results.items(), colors):
            plt.plot(
                r["alpha"], r["mean_mrr"], marker="o", linewidth=2, color=color,
                label=f"top {r['percent']:g}% (n={r['num_queries']})",
            )
        plt.xlabel("Alpha")
        plt.ylabel("Mean MRR")
        plt.title(
            f"Mean MRR vs Alpha by entropy percentile tier "
            f"({Experiment._UTILITY_FUNCTION_NAMES.get(self.utility_function, 'utility')})\n"
            f"{self.rag.dataset.__class__.__name__}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.0])
        comparison_plot_path = self.output_path / "mrr_vs_alpha_comparison.png"
        plt.savefig(str(comparison_plot_path), dpi=150, bbox_inches="tight")
        plt.close()

        self.save_results({"tiers": tier_results, "comparison_plot_file": str(comparison_plot_path)})
