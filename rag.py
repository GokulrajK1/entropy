import os
from abc import ABC, abstractmethod
import faiss
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import tiktoken
import nltk
import matplotlib
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor

# reduce log verbosity for various libraries
os.environ.setdefault("HF_HUB_DISABLE_WARNING", "1")  # hide unauthenticated HF hub warning
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
# optional: suppress matplotlib debug output
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# also configure transformers logging via its helper
from transformers import logging as _tf_logging
_tf_logging.set_verbosity_error()


from rag_datasets import RAGDataset, Query, Passage, Answer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
matplotlib.use("Agg")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

class EntropyCalculator(ABC):
    @abstractmethod
    def calculate_entropy(self, text : str) -> float:
        pass

class AutoregressiveEntropyCalculator(EntropyCalculator):
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def calculate_entropy_details(self, text, return_token_details=True):
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        token_entropies = []
        token_probs = []
        tokens = []

        for i in range(1, input_ids.shape[1]):
            position_logits = logits[0, i-1, :]
            probs = torch.softmax(position_logits, dim=0)
            entropy = -torch.sum(probs * torch.log2(probs + 1e-10))

            token_entropies.append(entropy.item())

            actual_token_id = input_ids[0, i].item()
            actual_token_prob = probs[actual_token_id].item()
            token_probs.append(actual_token_prob)

            token = self.tokenizer.decode([actual_token_id])
            tokens.append(token)

        predictive_entropy = np.mean(token_entropies) if token_entropies else 0.0
        perplexity = 2 ** predictive_entropy if token_entropies else float("inf")

        cross_entropy = -np.mean([np.log2(p) for p in token_probs]) if token_probs else 0.0
        avg_entropy = -np.mean(np.log(token_probs)) if token_probs else 0.0

        results = {
            'text': text,
            'num_tokens': len(tokens),
            'average_entropy': avg_entropy,
            'perplexity': perplexity,
            'cross_entropy': cross_entropy,
        }

        if return_token_details:
            results['tokens'] = tokens
            results['token_entropies'] = token_entropies
            results['token_probabilities'] = token_probs

        return results

    def calculate_entropy(self, text : str) -> float:
        metrics = self.calculate_entropy_details(text, return_token_details=False)
        val = float(metrics["average_entropy"])
        return val

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list[str]) -> np.ndarray:
    arr = model.encode(texts, convert_to_numpy=True).astype("float32")
    return arr

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens_tiktoken(text: str) -> int:
    n = len(enc.encode(text))
    return n

class Metric:
    def __init__(self, hit_recall : float = 0.0, overall_recall : float = 0.0, precision : float = 0.0, f1 : float = 0.0, mrr : float = 0.0, dcg : float = 0.0, recall_by_k : dict = {}) -> None:
        self.hit_recall = hit_recall
        self.overall_recall = overall_recall
        self.precision = precision
        self.f1 = f1
        self.mrr = mrr
        self.dcg = dcg
        self.recall_by_k = recall_by_k

class EntropyRAG:
    def __init__(self, dataset : RAGDataset, entropy_calculator : EntropyCalculator, embed_text_function, count_tokens_function) -> None:
        self.dataset = dataset
        self.entropy_calculator = entropy_calculator
        self.embed_text_function = embed_text_function
        self.count_tokens_function = count_tokens_function
        self.faiss_index_cache_path = dataset.data_path + "/faiss_index.pkl"
        self.index = None
        self.passage_ids : list[str] = []
        self.entropy_store : dict[str, float] = {}
        self.entropy_cache_path = os.path.join(dataset.data_path, "entropy_store.pkl")
        self.entropy_mean = None
        self.entropy_std = None
        self.query_entropy_store : dict[str, float] = {}
        self.query_entropy_cache_path = os.path.join(dataset.data_path, "query_entropy_store.pkl")
    
    def save_faiss_index(self):
        if self.index is None:
            raise ValueError("Index not built yet.")

        index_bytes = faiss.serialize_index(self.index)

        bundle = {"faiss_index": index_bytes, "passage_ids": self.passage_ids}

        with open(self.faiss_index_cache_path, "wb") as f:
            pickle.dump(bundle, f)

        print(f"Saved FAISS cache to {self.faiss_index_cache_path}")

    def load_faiss_index(self):
        if not os.path.exists(self.faiss_index_cache_path):
            return False

        with open(self.faiss_index_cache_path, "rb") as f:
            bundle = pickle.load(f)

        self.index = faiss.deserialize_index(bundle["faiss_index"])
        self.passage_ids = bundle["passage_ids"]

        return True
    
    def normalize_similarity_list(self, similarities):
        sims = np.array(similarities, dtype=np.float64)
        sims = (sims + 1.0) / 2.0
        sims = np.clip(sims, 0.0, 1.0)
        return sims.tolist()

    def create_faiss_index(self, batch_size: int = 4096, refresh : bool = False):

        if not self.dataset.passages:
            raise ValueError("Call dataset.load_passages() first.")
        
        if not refresh and self.load_faiss_index():
            print("FAISS index loaded from cache. Skipping index creation.")
            return
        
        print("Creating FAISS index...")

        self.index = None
        self.passage_ids = []
        total = len(self.dataset.passages)

        dim = None
        ids_batch = []
        texts_batch = []

        added = 0
        for i, (pid, passage) in enumerate(self.dataset.passages.items(), start=1):
            ids_batch.append(pid)
            texts_batch.append(passage.text)

            if len(texts_batch) >= batch_size or i == total:

                Xb = self.embed_text_function(texts_batch).astype("float32")
                faiss.normalize_L2(Xb)

                if self.index is None:
                    dim = Xb.shape[1]

                    self.index = faiss.IndexFlatIP(dim)

                self.index.add(Xb)
                self.passage_ids.extend(ids_batch)
                added += len(ids_batch)

                ids_batch = []
                texts_batch = []

                print(f"Processed {i}/{total} passages. Added {added} to index.")

        self.save_faiss_index()

    def fit_entropy_normalizer(self):
        values = np.array(list(self.entropy_store.values()), dtype=np.float64)

        self.entropy_mean = float(values.mean())
        self.entropy_std = float(values.std())

        if self.entropy_std < 1e-8:
            self.entropy_std = 1.0

    def normalize_entropy_list(self, entropy_values):
        ent = np.array(entropy_values, dtype=np.float64)
        z = (ent - self.entropy_mean) / self.entropy_std
        norm = 1.0 / (1.0 + np.exp(-z))
        return norm.tolist()

    def load_entropy_store(self) -> bool:
        if not os.path.exists(self.entropy_cache_path):
            return False

        with open(self.entropy_cache_path, "rb") as f:
            self.entropy_store = pickle.load(f)

        self.fit_entropy_normalizer()

        return True

    def save_entropy_store(self) -> None:
        with open(self.entropy_cache_path, "wb") as f:
            pickle.dump(self.entropy_store, f)

    def create_entropy_store(self, refresh : bool = False):

        if not self.dataset.passages:
            raise ValueError("Call dataset.load_passages() first.")
        
        if not refresh and self.load_entropy_store():
            print("Entropy store loaded from cache. Skipping creation.")
            return
        
        total = len(self.dataset.passages)
        for idx, (pid, passage) in enumerate(self.dataset.passages.items(), start=1):
            self.entropy_store[pid] = self.entropy_calculator.calculate_entropy(passage.text)
            if idx % 1000 == 0 or idx == total:
                print(f"Computed entropy for {idx}/{total} passages")

        self.fit_entropy_normalizer()
        
        self.save_entropy_store()

    def load_query_entropy_store(self) -> bool:
        if not os.path.exists(self.query_entropy_cache_path):
            return False

        with open(self.query_entropy_cache_path, "rb") as f:
            store = pickle.load(f)

        if not isinstance(store, dict):
            return False

        self.query_entropy_store = store
        return True

    def save_query_entropy_store(self) -> None:
        with open(self.query_entropy_cache_path, "wb") as f:
            pickle.dump(self.query_entropy_store, f)

    def create_query_entropy_store(self, refresh: bool = False):

        if not self.dataset.queries:
            raise ValueError("Call dataset.load_queries() first.")

        if refresh:
            self.query_entropy_store = {}
        else:
            self.load_query_entropy_store()

        missing_ids = [qid for qid in self.dataset.queries if qid not in self.query_entropy_store]
        if not missing_ids:
            print("Query entropy store loaded from cache. Skipping creation.")
            return

        total = len(missing_ids)
        for idx, qid in enumerate(missing_ids, start=1):
            self.query_entropy_store[qid] = self.entropy_calculator.calculate_entropy(self.dataset.queries[qid].text)
            if idx % 1000 == 0 or idx == total:
                print(f"Computed entropy for {idx}/{total} queries")

        self.save_query_entropy_store()

    def build_artifacts(self, refresh_entropy: bool = False, refresh_vectordb: bool = False, refresh_query_entropy: bool = False):
        if not self.dataset.passages:
            raise ValueError("Call dataset.load_passages() first.")

        with ThreadPoolExecutor(max_workers=3) as executor:
            faiss_future = executor.submit(self.create_faiss_index, refresh=refresh_vectordb)
            entropy_future = executor.submit(self.create_entropy_store, refresh=refresh_entropy)
            query_entropy_future = executor.submit(self.create_query_entropy_store, refresh=refresh_query_entropy)
            faiss_future.result()
            entropy_future.result()
            query_entropy_future.result()

    def retrieve_top_k_passages(self, query: Query, k: int) -> list[Passage]:
        if self.index is None:
            raise ValueError("Call create_faiss_index() first.")
        
        if self.entropy_store is None:
            raise ValueError("Call create_entropy_store() first.")

        qv = self.embed_text_function([query.text]).astype("float32")
        faiss.normalize_L2(qv)
        scores, indices = self.index.search(qv, k)

        candidate_passages_ids = [self.passage_ids[i] for i in indices[0] if i != -1]
        candidate_passages_scores = [float(s) for s in scores[0][:len(candidate_passages_ids)]]
        candidate_passages_entropies = []

        for pid in candidate_passages_ids:
            if pid not in self.entropy_store:
                raise ValueError(f"pid {pid} not found in the entropy store.")
            
            candidate_passages_entropies.append(self.entropy_store[pid])
                
        similiarity_scores_normalized = candidate_passages_scores
        entropy_values_normalized = candidate_passages_entropies

        return [Passage(pid, self.dataset.passages[pid].text, score, entropy) for pid, score, entropy in zip(candidate_passages_ids, similiarity_scores_normalized, entropy_values_normalized)]

    def evaluate_relevant_passages(self, recalled_passages_ids : list[str], query : Query, cutoffs : list[int] = None) -> Metric:
        if not self.dataset.answers:
            raise ValueError("Call dataset.load_answers() first.")

        relevant_passage_id_set = self.dataset.answers[query.id].relevant_passage_ids
        recalled_passage_id_set = set(recalled_passages_ids)

        if len(relevant_passage_id_set) == 0 or len(recalled_passage_id_set) == 0:
            return None

        total_positives = len(relevant_passage_id_set & recalled_passage_id_set)
        hit_recall = 1 if total_positives > 0 else 0
        overall_recall = total_positives / len(relevant_passage_id_set)
        precision = total_positives / len(recalled_passage_id_set) if recalled_passage_id_set else 0.0
        f1 = 0.0 if (precision + overall_recall) == 0 else 2 * (precision * overall_recall) / (precision + overall_recall)
        mrr = 0.0
        for rank, pid in enumerate(recalled_passages_ids, start=1):
            if pid in relevant_passage_id_set:
                mrr = 1.0 / rank
                break

        dcg = 0
        for rank, pid in enumerate(recalled_passage_id_set, start=1):
            if pid in relevant_passage_id_set:
                dcg += 1 / (np.log2(rank + 1))

        if cutoffs is None:
            cutoffs = []

        recall_by_k = {}
        
        for k in cutoffs:
            topk_passage_ids = recalled_passages_ids[:k]
            tp_k = len(set(topk_passage_ids) & relevant_passage_id_set)
            recall_by_k[k] = tp_k / len(relevant_passage_id_set)


        return Metric(hit_recall=hit_recall, overall_recall=overall_recall, precision=precision, f1=f1, mrr=mrr, dcg=dcg, recall_by_k=recall_by_k)

class UtilityFunction:
    U1 = lambda s, e, alpha : s * alpha + e * (1 - alpha)
    U2 = lambda s, e, alpha : s * (1 + alpha * e)
    # Same form as U2, but expects e pre-normalized to [0, 1] (see
    # experiment_helpers.normalized_entropy_lookup) so alpha controls a
    # comparable-magnitude blend instead of being swamped by raw entropy.
    U3 = lambda s, e, alpha : s * (1 + alpha * e)
    # Same form as U2/U3, but e is normalized to [0, 1] relative to the
    # per-query candidate set (see experiment_helpers.normalize_values).
    U4 = lambda s, e, alpha : s * (1 + alpha * e)