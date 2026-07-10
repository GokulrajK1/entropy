import argparse
import inspect
import json
import sys
from pathlib import Path

EXPERIMENT_CHOICES = [
    "simple_reordering",
    "reordering_recall",
    "similarity_entropy_table",
    "similarity_utility_table",
    "difficulty_mrr_comparison",
    "difficulty_alpha_sweep",
    "query_entropy_difficulty_mrr",
    "query_entropy_difficulty_split",
    "difficulty_ranking_complicated_alpha_sweep",
    "entropy_threshold_alpha_sweep",
    "entropy_threshold_top_passage_mrr",
    "ragbench_mrr_eval",
]

DATASET_CHOICES = ["msmacro", "ragbench_covidqa", "natural_questions", "hotpotqa"]
ENTROPY_CALCULATOR_CHOICES = ["gpt2"]
EMBEDDER_CHOICES = ["all-MiniLM-L6-v2"]
TOKEN_COUNTER_CHOICES = ["tiktoken"]
UTILITY_CHOICES = ["u1", "u2", "u3", "u4"]


def _resolve_path(value: str, base_dir: Path) -> str:
    path_value = Path(value)
    if path_value.is_absolute():
        return str(path_value)
    return str((base_dir / path_value).resolve())


def _load_experiment_param_names(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file_handle:
        config = json.load(file_handle)

    if isinstance(config, list):
        return [str(item).strip() for item in config if str(item).strip()]

    if isinstance(config, dict):
        params = config.get("params") or config.get("parameters") or config.get("argument_names")
        if isinstance(params, list):
            return [str(item).strip() for item in params if str(item).strip()]

    raise ValueError(f"Experiment config must be a JSON array of parameter names: {config_path}")


def _comma_separated_strings(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _comma_separated_floats(value):
    return [float(item) for item in _comma_separated_strings(value)]


def _build_parser(experiment_class=None, param_names=None):
    parser = argparse.ArgumentParser(
        description="Run experiments or refresh caches for RAG experiments",
        conflict_handler="resolve",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Run an experiment")
    group.add_argument("--refresh_entropy", action="store_true", help="Recompute entropy for a dataset and persist the cache")
    group.add_argument("--refresh_vectordb", action="store_true", help="Recreate the FAISS index for a dataset")

    parser.add_argument("--experiment", choices=EXPERIMENT_CHOICES, default="simple_reordering", help="Experiment to run (when --run is used)")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="msmacro", help="Dataset to use")
    parser.add_argument("--entropy_calculator", choices=ENTROPY_CALCULATOR_CHOICES, default="gpt2", help="Entropy calculator to use")
    parser.add_argument("--embedder", choices=EMBEDDER_CHOICES, default="all-MiniLM-L6-v2", help="Embedder to use")
    parser.add_argument("--token_counter", choices=TOKEN_COUNTER_CHOICES, default="tiktoken", help="Token counter to use")
    parser.add_argument("--utility", choices=UTILITY_CHOICES, default="u1", help="Default utility function selection")
    parser.add_argument("--query_file", type=str, default=None, help="Path to a TSV file containing query ids and questions. If provided, it overrides --query_ids")
    parser.add_argument("--params_json", type=str, default=None, help="Path to a JSON file containing experiment-specific parameter names")

    if experiment_class is not None and param_names is not None:
        init_signature = inspect.signature(experiment_class.__init__)
        for name in param_names:
            if name == "utility_function":
                parser.add_argument(
                    f"--{name}",
                    choices=UTILITY_CHOICES,
                    default=argparse.SUPPRESS,
                    help="Utility function selection to map to a callable",
                )
                continue

            parameter = init_signature.parameters.get(name)
            argument_kwargs = {
                "default": argparse.SUPPRESS,
            }

            if name == "query_ids":
                argument_kwargs["type"] = _comma_separated_strings
            elif name == "alphas":
                argument_kwargs["type"] = _comma_separated_floats
            elif parameter is not None:
                annotation = parameter.annotation
                default_value = parameter.default
                if annotation is int or isinstance(default_value, int):
                    argument_kwargs["type"] = int
                elif annotation is float or isinstance(default_value, float):
                    argument_kwargs["type"] = float
                elif annotation is list or isinstance(default_value, list):
                    argument_kwargs["type"] = _comma_separated_strings
                else:
                    argument_kwargs["type"] = str
            else:
                argument_kwargs["type"] = str

            parser.add_argument(f"--{name}", **argument_kwargs)

    return parser


def _load_runtime_dependencies():
    from experiments import (
        DifficultyAlphaSweepExperiment,
        DifficultyMRRComparisonExperiment,
        DifficultyRankingComplicatedAlphaSweepExperiment,
        EntropyThresholdAlphaSweepExperiment,
        EntropyThresholdTopPassageMRRExperiment,
        QueryEntropyDifficultyMRRExperiment,
        QueryEntropyDifficultySplitExperiment,
        RAGBenchMRREvaluationExperiment,
        ReorderingRecallExperiment,
        SimilarityEntropyTableExperiment,
        SimilarityUtilityTableExperiment,
        SimpleReorderingExperiment,
    )
    from rag import AutoregressiveEntropyCalculator, EntropyRAG, embed_texts, count_tokens_tiktoken, UtilityFunction
    from rag_datasets import MSMACRODataset, NaturalQuestionsDataset, RAGBenchCovidQADataset, HotpotQADataset

    dataset_map = {
        "msmacro": MSMACRODataset,
        "ragbench_covidqa": RAGBenchCovidQADataset,
        "natural_questions": NaturalQuestionsDataset,
        "hotpotqa": HotpotQADataset,
    }

    entropy_calc_map = {
        "gpt2": lambda: AutoregressiveEntropyCalculator(model_name="gpt2"),
    }

    embedder_map = {
        "all-MiniLM-L6-v2": lambda: embed_texts,
    }

    token_counter_map = {
        "tiktoken": lambda: count_tokens_tiktoken,
    }

    utility_function_map = {
        "u1": UtilityFunction.U1,
        "u2": UtilityFunction.U2,
        "u3": UtilityFunction.U3,
        "u4": UtilityFunction.U4,
    }

    experiment_map = {
        "simple_reordering": SimpleReorderingExperiment,
        "reordering_recall": ReorderingRecallExperiment,
        "similarity_entropy_table": SimilarityEntropyTableExperiment,
        "similarity_utility_table": SimilarityUtilityTableExperiment,
        "difficulty_mrr_comparison": DifficultyMRRComparisonExperiment,
        "difficulty_alpha_sweep": DifficultyAlphaSweepExperiment,
        "query_entropy_difficulty_mrr": QueryEntropyDifficultyMRRExperiment,
        "query_entropy_difficulty_split": QueryEntropyDifficultySplitExperiment,
        "difficulty_ranking_complicated_alpha_sweep": DifficultyRankingComplicatedAlphaSweepExperiment,
        "entropy_threshold_alpha_sweep": EntropyThresholdAlphaSweepExperiment,
        "entropy_threshold_top_passage_mrr": EntropyThresholdTopPassageMRRExperiment,
        "ragbench_mrr_eval": RAGBenchMRREvaluationExperiment,
    }

    return {
        "dataset_map": dataset_map,
        "entropy_calc_map": entropy_calc_map,
        "embedder_map": embedder_map,
        "token_counter_map": token_counter_map,
        "utility_function_map": utility_function_map,
        "experiment_map": experiment_map,
        "EntropyRAG": EntropyRAG,
    }


def main():
    base_parser = _build_parser()
    base_args, _ = base_parser.parse_known_args()

    runtime = _load_runtime_dependencies()
    dataset_map = runtime["dataset_map"]
    entropy_calc_map = runtime["entropy_calc_map"]
    embedder_map = runtime["embedder_map"]
    token_counter_map = runtime["token_counter_map"]
    utility_function_map = runtime["utility_function_map"]
    experiment_map = runtime["experiment_map"]
    EntropyRAG = runtime["EntropyRAG"]

    ExperimentClass = experiment_map.get(base_args.experiment)
    if ExperimentClass is None:
        print(f"Unknown experiment: {base_args.experiment}")
        sys.exit(1)

    experiment_signature = inspect.signature(ExperimentClass.__init__)

    config_path = Path(base_args.params_json) if base_args.params_json else Path("experiments") / base_args.experiment / "params.json"
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    experiment_param_names = _load_experiment_param_names(config_path)
    parser = _build_parser(ExperimentClass, experiment_param_names)
    args = parser.parse_args()

    DatasetClass = dataset_map.get(args.dataset)
    if DatasetClass is None:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    dataset = DatasetClass()
    dataset.load_all()

    entropy_calc_factory = entropy_calc_map.get(args.entropy_calculator)
    if entropy_calc_factory is None:
        print(f"Unknown entropy calculator: {args.entropy_calculator}")
        sys.exit(1)
    entropy_calc = entropy_calc_factory()

    embedder_factory = embedder_map.get(args.embedder)
    embedder = embedder_factory()

    token_counter_factory = token_counter_map.get(args.token_counter)
    token_counter = token_counter_factory()

    utility_function = utility_function_map.get(args.utility)
    if utility_function is None:
        print(f"Unknown utility function: {args.utility}")
        sys.exit(1)

    rag = EntropyRAG(dataset, entropy_calc, embedder, token_counter)

    if args.refresh_entropy:
        print(f"Refreshing entropy for dataset {args.dataset}...")
        rag.create_entropy_store(refresh=True)
        print("Entropy refresh complete.")
        return

    if args.refresh_vectordb:
        print(f"Refreshing FAISS index for dataset {args.dataset}...")
        rag.create_faiss_index(refresh=True)
        print("FAISS refresh complete.")
        return

    if not args.run:
        return

    rag.build_artifacts()

    experiment_kwargs = {}
    for name in experiment_param_names:
        if hasattr(args, name):
            value = getattr(args, name)
            if name == "utility_function":
                experiment_kwargs[name] = utility_function_map.get(value, value)
            else:
                experiment_kwargs[name] = value

    if "utility_function" in experiment_signature.parameters and "utility_function" not in experiment_kwargs:
        experiment_kwargs["utility_function"] = utility_function

    if args.query_file is not None:
        query_file_path = Path(args.query_file)
        if not query_file_path.is_absolute():
            query_file_path = Path.cwd() / query_file_path

        if not query_file_path.exists():
            print(f"Query file not found: {query_file_path}")
            sys.exit(1)

        query_ids = ExperimentClass.load_query_ids_from_file(query_file_path)
        if "query_ids" in experiment_signature.parameters:
            experiment_kwargs.setdefault("query_ids", query_ids)
        print(f"Loaded {len(query_ids)} query ids from {query_file_path}")

    if experiment_param_names:
        print(f"Loaded experiment parameters from {config_path}")
    else:
        print(f"No experiment parameters declared in {config_path}; running with constructor defaults.")

    exp = ExperimentClass(rag, **experiment_kwargs)
    print(f"Running experiment {args.experiment} on dataset {args.dataset}...")
    exp.run()
    print("Experiment finished.")


if __name__ == "__main__":
    main()
