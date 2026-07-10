import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np

from rag import UtilityFunction


class Experiment(ABC):
    RESULTS_DIRECTORY_NAME = "results"

    _UTILITY_FUNCTION_NAMES = {
        UtilityFunction.U1: "U1",
        UtilityFunction.U2: "U2",
        UtilityFunction.U3: "U3",
        UtilityFunction.U4: "U4",
    }

    @staticmethod
    def load_query_ids_from_file(query_file_path):
        query_ids = []
        with open(query_file_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if not stripped:
                    continue
                query_id = stripped.split("	", 1)[0].strip()
                if query_id:
                    query_ids.append(query_id)
        return query_ids

    @staticmethod
    def resolve_query_ids(dataset, query_ids=None, number_of_queries=None):
        all_query_ids = list(dataset.queries.keys())
        if query_ids is None:
            if number_of_queries is None:
                return all_query_ids
            return all_query_ids[:number_of_queries]

        query_index = {str(qid): qid for qid in all_query_ids}
        resolved = []
        missing = []
        for qid in query_ids:
            key = str(qid)
            if key in query_index:
                resolved.append(query_index[key])
            else:
                missing.append(qid)

        if missing:
            print(f"Warning: query ids not found in dataset and will be skipped: {missing}")
        return resolved

    def _start_run(self, experiment_file, parameters):
        """Create experiments/<name>/results/<timestamp>/ and record the run's parameters."""
        experiment_dir = Path(experiment_file).resolve().parent
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_path = experiment_dir / self.RESULTS_DIRECTORY_NAME / timestamp
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.parameters = self._json_safe(parameters)
        return self.output_path

    @classmethod
    def _json_safe(cls, value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): cls._json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(val) for val in value]
        if callable(value):
            return cls._UTILITY_FUNCTION_NAMES.get(value, getattr(value, "__name__", repr(value)))
        return value

    def save_results(self, results):
        payload = {"parameters": self.parameters, "results": self._json_safe(results)}
        output_file = self.output_path / "result.json"
        with open(output_file, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)
        return output_file

    @abstractmethod
    def run(self):
        pass
