from abc import ABC, abstractmethod
import json
import ir_datasets
import os
import random
import sys

class Query:
    def __init__(self, id, text) -> None:
        self.id = id
        self.text = text

class Passage:
    def __init__(self, id, text, score=None, entropy=None) -> None:
        self.id = id
        self.text = text
        self.similarity_score = score
        self.entropy = entropy

class Answer:
    def __init__(self, text : str = None, relevant_passage_ids : set[str] = None) -> None:
        self.text = text
        self.relevant_passage_ids = relevant_passage_ids if relevant_passage_ids is not None else set()

class RAGDataset(ABC):

    def __init__(self) -> None:
        self.queries: dict[str, Query] = {}
        self.passages: dict[str, Passage] = {}
        self.answers : dict[str, Answer] = {}

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def load_queries(self):
        pass

    @abstractmethod
    def load_passages(self):
        pass

    @abstractmethod
    def load_answers(self):
        pass

class MSMACRODataset(RAGDataset):

    def __init__(self, data_path: str = "DATA/MSMACRO"):
        super().__init__()
        self.data_path: str = data_path

    def download(self):
        os.makedirs(self.data_path, exist_ok=True)

        queries_path   = os.path.join(self.data_path, "queries.dev.small.tsv")
        qrels_path     = os.path.join(self.data_path, "qrels.dev.small.tsv")
        top1000_path   = os.path.join(self.data_path, "top1000.dev.small.tsv")
        collection_path= os.path.join(self.data_path, "collection.tsv")

        required_files = [queries_path, qrels_path, top1000_path, collection_path]
        if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in required_files):
            return

        ds_small = ir_datasets.load("msmarco-passage/dev/small")
        ds_docs  = ir_datasets.load("msmarco-passage")

        q_count = 0
        with open(queries_path, "w", encoding="utf-8") as f:
            for q in ds_small.queries_iter():
                clean_text = q.text.replace("\t", " ").strip()
                f.write(f"{q.query_id}\t{clean_text}\n")
                q_count += 1

        needed_doc_ids = set()
        qr_count = 0
        rel_pos = 0
        with open(qrels_path, "w", encoding="utf-8") as f:
            for qr in ds_small.qrels_iter():
                f.write(f"{qr.query_id}\t0\t{qr.doc_id}\t{qr.relevance}\n")
                qr_count += 1
                if int(qr.relevance) > 0:
                    needed_doc_ids.add(qr.doc_id)
                    rel_pos += 1
                
        sd_count = 0
        with open(top1000_path, "w", encoding="utf-8") as f:
            for sd in ds_small.scoreddocs_iter():
                f.write(f"{sd.query_id}\t{sd.doc_id}\t{sd.score}\n")
                needed_doc_ids.add(sd.doc_id)
                sd_count += 1
                
        store = ds_docs.docs_store()

        missing = 0
        wrote_docs = 0
   
        with open(collection_path, "w", encoding="utf-8") as f:
            for i, doc_id in enumerate(needed_doc_ids, start=1):
                doc = store.get(doc_id)
                if doc is None:
                    missing += 1
                    continue
                text = doc.text.replace("\t", " ").strip()
                f.write(f"{doc.doc_id}\t{text}\n")
                wrote_docs += 1

        def _size(p): return os.path.getsize(p) if os.path.exists(p) else -1

        print("Wrote:")
        print(" -", queries_path, _size(queries_path))
        print(" -", qrels_path, _size(qrels_path))
        print(" -", top1000_path, _size(top1000_path))
        print(" -", collection_path, _size(collection_path))
        if missing:
            print(f"Note: {missing} doc_ids missing from docs_store (usually tiny).")

    def load_queries(self) -> dict[str, Query]:
        path = os.path.join(self.data_path, "queries.dev.small.tsv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Queries file not found at {path}. Run download method first.")

        loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid, text = line.rstrip("\n").split("\t", 1)
                self.queries[qid] = Query(qid, text)
                loaded += 1

        return self.queries

    def load_passages(self) -> dict[str, Passage]:
        path = os.path.join(self.data_path, "collection.tsv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Queries file not found at {path}. Run download method first.")

        loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pid, text = line.rstrip("\n").split("\t", 1)
                self.passages[pid] = Passage(pid, text)
                loaded += 1

        return self.passages

    def load_answers(self) -> dict[str, Answer]:
        path = os.path.join(self.data_path, "qrels.dev.small.tsv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Queries file not found at {path}. Run download method first.")

        lines = 0
        pos = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid, _, pid, rel = line.rstrip("\n").split("\t")
                lines += 1
                if int(rel) > 0:
                    pos += 1
                    if qid not in self.answers:
                        self.answers[qid] = Answer(relevant_passage_ids=set([pid]))
                    else:
                        self.answers[qid].relevant_passage_ids.add(pid)

        return self.answers
    
    def load_all(self):
        self.load_queries()
        self.load_passages()
        self.load_answers()


class NaturalQuestionsDataset(RAGDataset):
    """Natural Questions retrieval set, sourced from BeIR/nq via ir_datasets.

    BeIR/nq packages NQ as a flat IR benchmark: a 2.68M-document Wikipedia
    passage corpus, ~3.5k queries, and qrels (avg ~1.2 relevant passages per
    query, all 3452 queries have at least one). The full corpus is too large
    to embed/score, so download() builds a local subset: all qrels-positive
    passages for the selected queries, padded with randomly sampled passages
    (doc IDs are sequential "doc0".."doc2681467", so sampling is a cheap
    docs_store() lookup, no full corpus scan) up to target_passage_pool.
    """

    MAX_PASSAGE_WORDS = 400  # keeps passages under GPT-2's 1024-token limit

    def __init__(self, data_path: str = "DATA/NATURAL_QUESTIONS", num_queries: int = 500, target_passage_pool: int = 5000, seed: int = 42):
        super().__init__()
        self.data_path: str = data_path
        self.num_queries: int = num_queries
        self.target_passage_pool: int = target_passage_pool
        self.seed: int = seed

    def _cached_paths(self):
        return {
            "queries": os.path.join(self.data_path, "queries.tsv"),
            "passages": os.path.join(self.data_path, "collection.tsv"),
            "answers": os.path.join(self.data_path, "qrels.tsv"),
        }

    def _truncate(self, text: str) -> str:
        words = text.split()
        if len(words) > self.MAX_PASSAGE_WORDS:
            words = words[:self.MAX_PASSAGE_WORDS]
        return " ".join(words).replace("\t", " ").strip()

    def download(self):
        os.makedirs(self.data_path, exist_ok=True)

        cached_paths = self._cached_paths()
        if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in cached_paths.values()):
            return

        ds = ir_datasets.load("beir/nq")

        qrels_by_query = {}
        for qrel in ds.qrels_iter():
            if qrel.relevance > 0:
                qrels_by_query.setdefault(qrel.query_id, set()).add(qrel.doc_id)

        selected_queries = []
        for query in ds.queries_iter():
            if query.query_id in qrels_by_query:
                selected_queries.append(query)
            if len(selected_queries) >= self.num_queries:
                break

        passage_ids = set()
        for query in selected_queries:
            passage_ids.update(qrels_by_query[query.query_id])

        store = ds.docs_store()
        docs_count = ds.docs_count()
        rng = random.Random(self.seed)
        while len(passage_ids) < self.target_passage_pool:
            passage_ids.add(f"doc{rng.randrange(docs_count)}")

        with open(cached_paths["queries"], "w", encoding="utf-8") as queries_f:
            for query in selected_queries:
                text = query.text.replace("\t", " ").strip()
                queries_f.write(f"{query.query_id}\t{text}\n")

        with open(cached_paths["passages"], "w", encoding="utf-8") as passages_f:
            for pid in passage_ids:
                doc = store.get(pid)
                text = self._truncate(doc.text)
                if doc.title:
                    text = f"{doc.title.strip()} {text}".strip()
                passages_f.write(f"{pid}\t{text}\n")

        qrel_count = 0
        with open(cached_paths["answers"], "w", encoding="utf-8") as answers_f:
            for query in selected_queries:
                for pid in qrels_by_query[query.query_id]:
                    answers_f.write(f"{query.query_id}\t0\t{pid}\t1\n")
                    qrel_count += 1

        print(f"Built Natural Questions (BeIR/nq) cache: {len(selected_queries)} queries, {len(passage_ids)} passages, {qrel_count} qrels")

    def load_queries(self) -> dict[str, Query]:
        path = self._cached_paths()["queries"]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Queries file not found at {path}. Run download method first.")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid, text = line.rstrip("\n").split("\t", 1)
                self.queries[qid] = Query(qid, text)

        return self.queries

    def load_passages(self) -> dict[str, Passage]:
        path = self._cached_paths()["passages"]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Passages file not found at {path}. Run download method first.")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pid, text = line.rstrip("\n").split("\t", 1)
                self.passages[pid] = Passage(pid, text)

        return self.passages

    def load_answers(self) -> dict[str, Answer]:
        path = self._cached_paths()["answers"]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Qrels file not found at {path}. Run download method first.")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid, _, pid, rel = line.rstrip("\n").split("\t")
                if int(rel) > 0:
                    if qid not in self.answers:
                        self.answers[qid] = Answer(relevant_passage_ids={pid})
                    else:
                        self.answers[qid].relevant_passage_ids.add(pid)

        return self.answers

    def load_all(self):
        self.download()
        self.load_queries()
        self.load_passages()
        self.load_answers()


class RAGBenchCovidQADataset(RAGDataset):
    """
    RAGBench dataset wrapper for CovidQA.
    Downloads and manages the CovidQA dataset from HuggingFace (RAGBench).
    Each query has exactly 3 relevant passages associated with it.
    
    Supports loading from:
    1. Local JSONL files (recommended for offline use)
    2. HuggingFace datasets library (if available)
    """

    def __init__(self, data_path: str = "DATA/RAGBENCH_COVIDQA", dataset_name: str = "covidqa"):
        super().__init__()
        self.data_path: str = data_path
        self.dataset_name: str = dataset_name
        self.hf_dataset = None

    def _preferred_split(self) -> str:
        """Return the dataset split to use, preferring train for CovidQA."""
        for split in ("train", "validation", "dev", "test"):
            if self.hf_dataset is not None and split in self.hf_dataset:
                return split
        if self.hf_dataset is not None:
            return list(self.hf_dataset.keys())[0]
        return "train"

    def download(self):
        """Download and cache the dataset from local files or HuggingFace."""
        os.makedirs(self.data_path, exist_ok=True)
        
        # First try to load from local JSONL files (preferred method)
        if self._try_load_from_local_jsonl():
            return
        
        # Fall back to HuggingFace datasets
        self._try_load_from_huggingface()
    
    def _try_load_from_local_jsonl(self) -> bool:
        """Try to load from local JSONL files. Returns True if successful."""
        
        queries_file = os.path.join(self.data_path, "queries.jsonl")
        passages_file = os.path.join(self.data_path, "passages.jsonl")
        qrels_file = os.path.join(self.data_path, "qrels.jsonl")
        
        if not all(os.path.exists(f) for f in [queries_file, passages_file, qrels_file]):
            return False
        
        print(f"Loading from local JSONL files in {self.data_path}/...")
        
        try:
            import json
            
            # Load queries
            queries_data = {}
            with open(queries_file, 'r', encoding='utf-8') as f:
                for line in f:
                    q = json.loads(line.strip())
                    queries_data[q['query_id']] = q
            
            # Load passages
            passages_data = {}
            with open(passages_file, 'r', encoding='utf-8') as f:
                for line in f:
                    p = json.loads(line.strip())
                    passages_data[p['passage_id']] = p
            
            # Load qrels
            qrels_data = {}
            with open(qrels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    qr = json.loads(line.strip())
                    qrels_data.setdefault(qr['query_id'], []).append(qr)
            
            # Format as HuggingFace-like dataset
            self.hf_dataset = {
                'train': [{
                    'query_id': qid,
                    'query': q['query'],
                    'contexts': [
                        {
                            'id': qr['passage_id'],
                            'text': passages_data[qr['passage_id']]['text'],
                            'is_relevant': qr.get('relevance', 0) > 0
                        }
                        for qr in qrels_data.get(qid, [])
                    ]
                } for qid, q in queries_data.items()]
            }
            
            print(f"✓ Successfully loaded from local JSONL files")
            print(f"  - {len(queries_data)} queries")
            print(f"  - {len(passages_data)} passages")
            print(f"  - {len(qrels_data)} queries with relevance judgments")
            return True
            
        except Exception as e:
            print(f"✗ Error loading from JSONL files: {e}")
            return False
    
    def _try_load_from_huggingface(self) -> bool:
        """Try to load from HuggingFace datasets library and cache to local files."""
        
        try:
            import importlib
            hf_datasets = importlib.import_module('datasets')
            load_dataset = hf_datasets.load_dataset
        except ImportError:
            print("HuggingFace datasets library not installed")
            return False
        
        print(f"Loading {self.dataset_name} from HuggingFace...")
        
        try:
            # Try the exact Hugging Face dataset ID first, then a few fallback aliases.
            dataset_paths = [
                ("galileo-ai/ragbench", self.dataset_name),  # exact path exposed by datasets-server
                ("ragbench", self.dataset_name),              # legacy alias
                (f"RAGBench/{self.dataset_name}", None),      # direct path guess
                (self.dataset_name, None),                     # bare config/name guess
            ]
            
            for dataset_path, config in dataset_paths:
                try:
                    if config:
                        self.hf_dataset = load_dataset(dataset_path, config)
                    else:
                        self.hf_dataset = load_dataset(dataset_path)
                    print(f"✓ Successfully loaded from HuggingFace: {dataset_path}")
                    
                    # Cache to local JSONL files for future runs
                    self._save_to_local_jsonl()
                    
                    return True
                except Exception as e:
                    continue
            
            print(f"✗ Could not find dataset '{self.dataset_name}' on HuggingFace")
            print(f"\n   To use your own data, provide JSONL files in:")
            print(f"   {self.data_path}/")
            print(f"   Required files:")
            print(f"     - queries.jsonl: {{'query_id': 'q_0', 'query': 'What is COVID?'}}")
            print(f"     - passages.jsonl: {{'passage_id': 'p_0', 'text': 'COVID is...'}}")
            print(f"     - qrels.jsonl: {{'query_id': 'q_0', 'passage_id': 'p_0', 'relevance': 1}}")
            return False
            
        except Exception as e:
            print(f"✗ Error loading from HuggingFace: {e}")
            return False
    
    def _save_to_local_jsonl(self):
        """Save HuggingFace dataset to local JSONL files for caching."""
        
        if self.hf_dataset is None:
            return
        
        try:
            import json
            
            # Cache the preferred split first; for CovidQA that is train.
            split = self._preferred_split()
            
            dataset_split = self.hf_dataset[split]
            
            queries_file = os.path.join(self.data_path, "queries.jsonl")
            passages_file = os.path.join(self.data_path, "passages.jsonl")
            qrels_file = os.path.join(self.data_path, "qrels.jsonl")
            
            # Collect unique passages by text so repeated documents across rows share the same ID.
            passages_dict = {}
            passage_text_to_id = {}
            qrels_list = []
            queries_list = []

            def sentence_keys_for_document(document_sentences):
                keys = []
                for sentence in document_sentences or []:
                    if isinstance(sentence, (list, tuple)) and sentence:
                        keys.append(str(sentence[0]))
                return keys
            
            for example in dataset_split:
                query_id = example.get('query_id', example.get('id', None))
                query_text = example.get('query', example.get('question', ''))
                relevant_sentence_keys = set(example.get('all_relevant_sentence_keys', []))
                
                if query_id and query_text:
                    queries_list.append({'query_id': query_id, 'query': query_text})
                
                # Process contexts/passages from either schema.
                passages = example.get('contexts', example.get('passages', example.get('documents', [])))
                documents_sentences = example.get('documents_sentences', [])
                if isinstance(passages, list):
                    for i, passage in enumerate(passages):
                        if isinstance(passage, dict):
                            passage_text = passage.get('text', passage.get('passage', ''))
                        else:
                            passage_text = str(passage)

                        if not passage_text:
                            continue

                        passage_id = passage_text_to_id.get(passage_text)
                        if passage_id is None:
                            passage_id = passage.get('id') if isinstance(passage, dict) else None
                            if not passage_id:
                                passage_id = f"{query_id}_p_{i}"
                            passage_text_to_id[passage_text] = passage_id
                            passages_dict[passage_id] = passage_text

                        passage_sentence_keys = sentence_keys_for_document(documents_sentences[i] if i < len(documents_sentences) else [])
                        is_relevant = any(key in relevant_sentence_keys for key in passage_sentence_keys)

                        qrels_list.append({
                            'query_id': query_id,
                            'passage_id': passage_id,
                            'relevance': 1 if is_relevant else 0
                        })
            
            # Write queries.jsonl
            with open(queries_file, 'w', encoding='utf-8') as f:
                for q in queries_list:
                    f.write(json.dumps(q) + '\n')
            
            # Write passages.jsonl
            with open(passages_file, 'w', encoding='utf-8') as f:
                for pid, text in passages_dict.items():
                    f.write(json.dumps({'passage_id': pid, 'text': text}) + '\n')
            
            # Write qrels.jsonl
            with open(qrels_file, 'w', encoding='utf-8') as f:
                for qr in qrels_list:
                    f.write(json.dumps(qr) + '\n')
            
            print(f"✓ Cached to local JSONL files:")
            print(f"  - {queries_file}")
            print(f"  - {passages_file}")
            print(f"  - {qrels_file}")
            
        except Exception as e:
            print(f"⚠ Warning: Could not cache to local JSONL files: {e}")
            # Don't fail - data is still loaded in memory

    def load_queries(self) -> dict[str, Query]:
        """Extract queries from the HuggingFace dataset."""
        if self.hf_dataset is None:
            self.download()
        
        split = self._preferred_split()
        dataset_split = self.hf_dataset[split]
        
        for example in dataset_split:
            # Get query text from the example
            query_text = example.get('query', example.get('question', ''))
            query_id = example.get('query_id', example.get('id', str(len(self.queries))))
            
            if query_text:
                self.queries[query_id] = Query(query_id, query_text)
        
        print(f"Loaded {len(self.queries)} queries")
        return self.queries

    def load_passages(self) -> dict[str, Passage]:
        """Extract passages from the HuggingFace dataset or local JSONL files."""
        if self.hf_dataset is None:
            self.download()
        
        # First, try to load from local passages.jsonl to preserve IDs
        passages_file = os.path.join(self.data_path, "passages.jsonl")
        if os.path.exists(passages_file):
            try:
                import json
                with open(passages_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        p = json.loads(line.strip())
                        passage_id = p.get('passage_id')
                        passage_text = p.get('text', '')
                        if passage_id and passage_text:
                            self.passages[passage_id] = Passage(passage_id, passage_text)
                print(f"Loaded {len(self.passages)} passages from local JSONL")
                return self.passages
            except Exception as e:
                print(f"⚠ Warning: Could not load passages from JSONL: {e}, falling back to HuggingFace")
        
        # Fallback: Extract passages from HuggingFace dataset
        split = self._preferred_split()
        dataset_split = self.hf_dataset[split]
        
        passage_text_to_id = {}  # Map text to consistent IDs
        
        for example in dataset_split:
            # Get passages/contexts from the example
            contexts = example.get('contexts', example.get('passages', example.get('documents', [])))
            
            if isinstance(contexts, list):
                for i, context in enumerate(contexts):
                    if isinstance(context, dict):
                        passage_text = context.get('text', context.get('passage', ''))
                        passage_id = context.get('id')
                    else:
                        # context is a string
                        passage_text = str(context)
                        passage_id = None
                    
                    if not passage_text:
                        continue
                    
                    # Use text-based deduplication: if we've seen this text, reuse its ID
                    if passage_text in passage_text_to_id:
                        passage_id = passage_text_to_id[passage_text]
                    else:
                        # Generate ID if not provided
                        if not passage_id:
                            passage_id = f"passage_{len(passage_text_to_id)}"
                        passage_text_to_id[passage_text] = passage_id
                        self.passages[passage_id] = Passage(passage_id, passage_text)
        
        print(f"Loaded {len(self.passages)} unique passages from HuggingFace")
        return self.passages

    def load_answers(self) -> dict[str, Answer]:
        """Extract answer/relevance mappings from local JSONL or HuggingFace dataset."""
        if self.hf_dataset is None:
            self.download()
        
        # First, try to load from local qrels.jsonl to preserve consistency
        qrels_file = os.path.join(self.data_path, "qrels.jsonl")
        if os.path.exists(qrels_file):
            try:
                import json
                with open(qrels_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        qr = json.loads(line.strip())
                        query_id = qr.get('query_id')
                        passage_id = qr.get('passage_id')
                        relevance = qr.get('relevance', 0)
                        
                        if query_id and passage_id and relevance > 0:
                            if query_id not in self.answers:
                                self.answers[query_id] = Answer(relevant_passage_ids=set([passage_id]))
                            else:
                                self.answers[query_id].relevant_passage_ids.add(passage_id)
                print(f"Loaded answers for {len(self.answers)} queries from local JSONL")
                return self.answers
            except Exception as e:
                print(f"⚠ Warning: Could not load answers from JSONL: {e}, falling back to HuggingFace")
        
        # Fallback: Extract from HuggingFace dataset
        split = self._preferred_split()
        dataset_split = self.hf_dataset[split]
        
        for example in dataset_split:
            query_id = example.get('query_id', example.get('id', None))
            if query_id is None:
                continue
            
            # Get relevant passages/contexts
            contexts = example.get('contexts', example.get('passages', example.get('documents', [])))
            relevant_passage_ids = set()
            relevant_sentence_keys = set(example.get('all_relevant_sentence_keys', []))
            documents_sentences = example.get('documents_sentences', [])
            
            if isinstance(contexts, list):
                for i, context in enumerate(contexts):
                    if isinstance(context, dict):
                        passage_text = context.get('text', context.get('passage', ''))
                        passage_id = context.get('id', f"passage_{i}")
                    else:
                        # context is a string - treat as relevant
                        passage_text = str(context)
                        passage_id = f"passage_{i}"

                    passage_sentence_keys = []
                    if i < len(documents_sentences):
                        for sentence in documents_sentences[i] or []:
                            if isinstance(sentence, (list, tuple)) and sentence:
                                passage_sentence_keys.append(str(sentence[0]))

                    is_relevant = any(key in relevant_sentence_keys for key in passage_sentence_keys)
                    
                    if is_relevant and passage_text:
                        # Find the passage_id in our passages dict
                        for pid, passage in self.passages.items():
                            if passage.text == passage_text:
                                relevant_passage_ids.add(pid)
                                break
            
            if relevant_passage_ids or contexts:
                # If no explicitly relevant passages but we have contexts,
                # treat them all as relevant (RAGBench typically provides relevant docs)
                if not relevant_passage_ids:
                    for context in contexts:
                        if isinstance(context, dict):
                            passage_text = context.get('text', context.get('passage', ''))
                        else:
                            passage_text = str(context)
                        
                        for pid, passage in self.passages.items():
                            if passage.text == passage_text:
                                relevant_passage_ids.add(pid)
                                break
                
                self.answers[query_id] = Answer(relevant_passage_ids=relevant_passage_ids)
        
        print(f"Loaded answers for {len(self.answers)} queries from HuggingFace")
        return self.answers
    
    def load_all(self):
        """Load all dataset components."""
        self.load_queries()
        self.load_passages()
        self.load_answers()


class HotpotQADataset(RAGDataset):
    """HotpotQA retrieval benchmark, sourced from beir/hotpotqa/train via ir_datasets.

    beir/hotpotqa/train provides ~85K multi-hop questions, each linked to two
    Wikipedia passages as supporting facts (qrels relevance=1). The full corpus
    has sparse Wikipedia curid-style doc IDs, so negative padding uses a single
    reservoir-sampling pass over docs_iter() rather than random integer sampling.

    download() builds a local TSV cache: all qrels-positive passages for the
    selected queries, padded with reservoir-sampled negatives up to
    target_passage_pool. When num_queries=None, all available queries are used.
    Passage text is truncated to MAX_PASSAGE_WORDS (~400 words) to stay within
    GPT-2's 1024-token limit.
    """

    MAX_PASSAGE_WORDS = 400

    def __init__(self, data_path: str = "DATA/HOTPOTQA", num_queries: int = None, target_passage_pool: int = 500000, seed: int = 42):
        super().__init__()
        self.data_path: str = data_path
        self.num_queries: int = num_queries
        self.target_passage_pool: int = target_passage_pool
        self.seed: int = seed

    def _cached_paths(self):
        return {
            "queries": os.path.join(self.data_path, "queries.tsv"),
            "passages": os.path.join(self.data_path, "collection.tsv"),
            "answers": os.path.join(self.data_path, "qrels.tsv"),
        }

    def _truncate(self, text: str) -> str:
        words = text.split()
        if len(words) > self.MAX_PASSAGE_WORDS:
            words = words[:self.MAX_PASSAGE_WORDS]
        return " ".join(words).replace("\t", " ").strip()

    def download(self):
        os.makedirs(self.data_path, exist_ok=True)

        cached = self._cached_paths()
        if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in cached.values()):
            return

        ds = ir_datasets.load("beir/hotpotqa/train")

        # Build qrels index: query_id -> set of relevant doc_ids
        qrels_by_query = {}
        for qrel in ds.qrels_iter():
            if qrel.relevance > 0:
                qrels_by_query.setdefault(qrel.query_id, set()).add(qrel.doc_id)

        # Select queries that have qrels, optionally capped by num_queries
        selected_queries = []
        for query in ds.queries_iter():
            if query.query_id in qrels_by_query:
                selected_queries.append(query)
            if self.num_queries is not None and len(selected_queries) >= self.num_queries:
                break

        # Collect all qrels-positive doc_ids for selected queries
        positive_ids = set()
        for query in selected_queries:
            positive_ids.update(qrels_by_query[query.query_id])

        passage_ids_ordered = list(positive_ids)

        # Pad with reservoir-sampled negatives if needed (doc IDs are sparse curids)
        need = self.target_passage_pool - len(passage_ids_ordered)
        if need > 0:
            rng = random.Random(self.seed)
            reservoir = []
            seen_positive = positive_ids
            for i, doc in enumerate(ds.docs_iter()):
                if doc.doc_id in seen_positive:
                    continue
                if len(reservoir) < need:
                    reservoir.append(doc.doc_id)
                    if len(reservoir) == need:
                        # Reservoir is full; no need to scan further
                        break
                else:
                    j = rng.randint(0, i)
                    if j < need:
                        reservoir[j] = doc.doc_id
                if i % 100000 == 0 and i > 0:
                    print(f"Reservoir sampling: scanned {i} docs, have {len(reservoir)}/{need} negatives")
            passage_ids_ordered.extend(reservoir)

        passage_id_set = set(passage_ids_ordered)

        # Write queries TSV
        with open(cached["queries"], "w", encoding="utf-8") as f:
            for query in selected_queries:
                text = query.text.replace("\t", " ").strip()
                f.write(f"{query.query_id}\t{text}\n")

        # Write passages TSV using docs_store for efficient point lookups
        store = ds.docs_store()
        wrote = 0
        with open(cached["passages"], "w", encoding="utf-8") as f:
            for pid in passage_ids_ordered:
                doc = store.get(pid)
                if doc is None:
                    continue
                text = self._truncate(doc.text)
                if hasattr(doc, "title") and doc.title:
                    text = f"{doc.title.strip()} {text}".strip()
                f.write(f"{pid}\t{text}\n")
                wrote += 1

        # Write qrels TSV
        qrel_count = 0
        with open(cached["answers"], "w", encoding="utf-8") as f:
            for query in selected_queries:
                for pid in qrels_by_query[query.query_id]:
                    if pid in passage_id_set:
                        f.write(f"{query.query_id}\t0\t{pid}\t1\n")
                        qrel_count += 1

        print(f"Built HotpotQA (beir/hotpotqa/train) cache: {len(selected_queries)} queries, {wrote} passages, {qrel_count} qrels")

    def load_queries(self) -> dict[str, Query]:
        path = self._cached_paths()["queries"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Queries file not found at {path}. Run download() first.")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid, text = line.rstrip("\n").split("\t", 1)
                self.queries[qid] = Query(qid, text)
        return self.queries

    def load_passages(self) -> dict[str, Passage]:
        path = self._cached_paths()["passages"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Passages file not found at {path}. Run download() first.")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pid, text = line.rstrip("\n").split("\t", 1)
                self.passages[pid] = Passage(pid, text)
        return self.passages

    def load_answers(self) -> dict[str, Answer]:
        path = self._cached_paths()["answers"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Qrels file not found at {path}. Run download() first.")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid, _, pid, rel = line.rstrip("\n").split("\t")
                if int(rel) > 0:
                    if qid not in self.answers:
                        self.answers[qid] = Answer(relevant_passage_ids={pid})
                    else:
                        self.answers[qid].relevant_passage_ids.add(pid)
        return self.answers

    def load_all(self):
        self.download()
        self.load_queries()
        self.load_passages()
        self.load_answers()

class FinQADataset(RAGDataset):
    """FinQA financial QA benchmark, sourced from GitHub czyssrs/FinQA raw JSON.

    Each example pairs a question with a bounded financial report excerpt
    (pre_text + linearized table + post_text) keyed by `filename`; excerpts
    are shared across ~3 questions on average (2110 unique excerpts / 6251
    train examples). download() combines train/dev/test (8281 examples total)
    into one query per example and one passage per unique excerpt, with 1:1
    query->passage qrels matching how questions were authored against a
    bounded excerpt.
    """

    MAX_PASSAGE_WORDS = 400

    SPLIT_URLS = {
        "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
        "dev": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
        "test": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
    }

    def __init__(self, data_path: str = "DATA/FIN_QA"):
        super().__init__()
        self.data_path: str = data_path

    def _cached_paths(self):
        return {
            "queries": os.path.join(self.data_path, "queries.tsv"),
            "passages": os.path.join(self.data_path, "collection.tsv"),
            "answers": os.path.join(self.data_path, "qrels.tsv"),
        }

    def _truncate(self, text: str) -> str:
        words = text.split()
        if len(words) > self.MAX_PASSAGE_WORDS:
            words = words[:self.MAX_PASSAGE_WORDS]
        return " ".join(words).replace("\t", " ").strip()

    def _linearize_table(self, table) -> str:
        rows = []
        for row in table or []:
            cells = [str(cell).strip() for cell in row if str(cell).strip()]
            if cells:
                rows.append(", ".join(cells))
        return " ; ".join(rows)

    def download(self):
        os.makedirs(self.data_path, exist_ok=True)

        cached = self._cached_paths()
        if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in cached.values()):
            return

        import urllib.request

        passages_by_id = {}
        query_rows = []
        qrel_rows = []

        for split, url in self.SPLIT_URLS.items():
            with urllib.request.urlopen(url) as response:
                examples = json.loads(response.read().decode("utf-8"))

            for example in examples:
                context_id = example["filename"]
                if context_id not in passages_by_id:
                    pre = " ".join(example.get("pre_text", []))
                    post = " ".join(example.get("post_text", []))
                    table_text = self._linearize_table(example.get("table", []))
                    full_text = f"{pre} {table_text} {post}".strip()
                    passages_by_id[context_id] = self._truncate(full_text)

                qa = example.get("qa")
                if not qa or not qa.get("question"):
                    continue

                qid = f"{split}_{example['id']}"
                question_text = qa["question"].replace("\t", " ").strip()
                query_rows.append((qid, question_text))
                qrel_rows.append((qid, context_id))

        with open(cached["passages"], "w", encoding="utf-8") as f:
            for pid, text in passages_by_id.items():
                f.write(f"{pid}\t{text}\n")

        with open(cached["queries"], "w", encoding="utf-8") as f:
            for qid, text in query_rows:
                f.write(f"{qid}\t{text}\n")

        with open(cached["answers"], "w", encoding="utf-8") as f:
            for qid, pid in qrel_rows:
                f.write(f"{qid}\t0\t{pid}\t1\n")

        print(f"Built FinQA cache: {len(query_rows)} queries, {len(passages_by_id)} passages, {len(qrel_rows)} qrels")

    def load_queries(self) -> dict[str, Query]:
        path = self._cached_paths()["queries"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Queries file not found at {path}. Run download() first.")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid, text = line.rstrip("\n").split("\t", 1)
                self.queries[qid] = Query(qid, text)
        return self.queries

    def load_passages(self) -> dict[str, Passage]:
        path = self._cached_paths()["passages"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Passages file not found at {path}. Run download() first.")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pid, text = line.rstrip("\n").split("\t", 1)
                self.passages[pid] = Passage(pid, text)
        return self.passages

    def load_answers(self) -> dict[str, Answer]:
        path = self._cached_paths()["answers"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Qrels file not found at {path}. Run download() first.")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid, _, pid, rel = line.rstrip("\n").split("\t")
                if int(rel) > 0:
                    if qid not in self.answers:
                        self.answers[qid] = Answer(relevant_passage_ids={pid})
                    else:
                        self.answers[qid].relevant_passage_ids.add(pid)
        return self.answers

    def load_all(self):
        self.download()
        self.load_queries()
        self.load_passages()
        self.load_answers()


class PubMedQADataset(RAGDataset):
    """PubMedQA biomedical QA benchmark, sourced from HuggingFace qiaojin/PubMedQA.

    Each example pairs a research question with the PubMed abstract it was
    written against (identified by pubid) -- a natural 1:1 qrel, no chunking
    needed since the abstract itself is the passage. download() combines the
    PQA-labeled (1000, human-annotated) and PQA-unlabeled (~61200, naturally
    authored but unlabeled) splits by default for max query variation;
    PQA-artificial (211K, template-generated from titles) is available via
    include_artificial=True but off by default since its query naturalness
    is weaker.
    """

    MAX_PASSAGE_WORDS = 400

    CONFIGS = {
        "labeled": "pqa_labeled",
        "unlabeled": "pqa_unlabeled",
        "artificial": "pqa_artificial",
    }

    def __init__(self, data_path: str = "DATA/PUBMED_QA", include_artificial: bool = False):
        super().__init__()
        self.data_path: str = data_path
        self.include_artificial: bool = include_artificial

    def _cached_paths(self):
        return {
            "queries": os.path.join(self.data_path, "queries.tsv"),
            "passages": os.path.join(self.data_path, "collection.tsv"),
            "answers": os.path.join(self.data_path, "qrels.tsv"),
        }

    def _truncate(self, text: str) -> str:
        words = text.split()
        if len(words) > self.MAX_PASSAGE_WORDS:
            words = words[:self.MAX_PASSAGE_WORDS]
        return " ".join(words).replace("\t", " ").strip()

    def download(self):
        os.makedirs(self.data_path, exist_ok=True)

        cached = self._cached_paths()
        if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in cached.values()):
            return

        import importlib
        hf_datasets = importlib.import_module("datasets")
        load_dataset = hf_datasets.load_dataset

        configs = ["labeled", "unlabeled"]
        if self.include_artificial:
            configs.append("artificial")

        passages_by_id = {}
        queries_by_id = {}

        for config_key in configs:
            split = load_dataset("qiaojin/PubMedQA", self.CONFIGS[config_key], split="train")
            for example in split:
                pubid = str(example["pubid"])
                question_text = (example.get("question") or "").replace("\t", " ").strip()
                if not question_text:
                    continue

                if pubid not in passages_by_id:
                    contexts = (example.get("context") or {}).get("contexts") or []
                    abstract_text = self._truncate(" ".join(contexts))
                    if not abstract_text:
                        continue
                    passages_by_id[pubid] = abstract_text

                queries_by_id[pubid] = question_text

        with open(cached["passages"], "w", encoding="utf-8") as f:
            for pid, text in passages_by_id.items():
                f.write(f"{pid}\t{text}\n")

        with open(cached["queries"], "w", encoding="utf-8") as f:
            for qid, text in queries_by_id.items():
                f.write(f"{qid}\t{text}\n")

        qrel_count = 0
        with open(cached["answers"], "w", encoding="utf-8") as f:
            for qid in queries_by_id:
                if qid in passages_by_id:
                    f.write(f"{qid}\t0\t{qid}\t1\n")
                    qrel_count += 1

        print(f"Built PubMedQA cache: {len(queries_by_id)} queries, {len(passages_by_id)} passages, {qrel_count} qrels")

    def load_queries(self) -> dict[str, Query]:
        path = self._cached_paths()["queries"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Queries file not found at {path}. Run download() first.")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid, text = line.rstrip("\n").split("\t", 1)
                self.queries[qid] = Query(qid, text)
        return self.queries

    def load_passages(self) -> dict[str, Passage]:
        path = self._cached_paths()["passages"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Passages file not found at {path}. Run download() first.")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pid, text = line.rstrip("\n").split("\t", 1)
                self.passages[pid] = Passage(pid, text)
        return self.passages

    def load_answers(self) -> dict[str, Answer]:
        path = self._cached_paths()["answers"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Qrels file not found at {path}. Run download() first.")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid, _, pid, rel = line.rstrip("\n").split("\t")
                if int(rel) > 0:
                    if qid not in self.answers:
                        self.answers[qid] = Answer(relevant_passage_ids={pid})
                    else:
                        self.answers[qid].relevant_passage_ids.add(pid)
        return self.answers

    def load_all(self):
        self.download()
        self.load_queries()
        self.load_passages()
        self.load_answers()


class QasperDataset(RAGDataset):
    """Qasper NLP-paper QA benchmark, sourced from HuggingFace allenai/qasper.

    Loads via revision="refs/convert/parquet" since the dataset repo ships a
    loading script that recent `datasets` versions reject for direct use.
    Each paper's full_text is already segmented into paragraphs -- the
    natural passage unit here. Each question's answers list evidence strings
    (verbatim paragraph text, or "FLOAT SELECTED" placeholders for
    table/figure evidence which are filtered out); download() exact-matches
    each evidence string back to its paragraph to build genuine
    paragraph-level qrels. Unanswerable questions and questions whose
    evidence doesn't match any paragraph are dropped. Passages are all
    paragraphs from papers that retain at least one usable question (not
    just the cited ones).
    """

    MAX_PASSAGE_TOKENS = 1000  # margin below GPT-2's 1024-position limit
    SPLITS = ["train", "validation", "test"]

    _tokenizer = None

    def __init__(self, data_path: str = "DATA/QASPER"):
        super().__init__()
        self.data_path: str = data_path

    def _cached_paths(self):
        return {
            "queries": os.path.join(self.data_path, "queries.tsv"),
            "passages": os.path.join(self.data_path, "collection.tsv"),
            "answers": os.path.join(self.data_path, "qrels.tsv"),
        }

    @classmethod
    def _get_tokenizer(cls):
        if cls._tokenizer is None:
            from transformers import GPT2TokenizerFast
            cls._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        return cls._tokenizer

    def _truncate(self, text: str) -> str:
        # Word-count truncation isn't safe here: Qasper's technical/scientific
        # text (citations, LaTeX-like notation, dense terminology) has a much
        # higher word-to-token ratio than general prose, so a 400-word cap can
        # still exceed GPT-2's 1024-token position limit and crash entropy scoring.
        text = " ".join(text.split())  # normalize all whitespace (incl. embedded newlines/tabs) to single spaces
        token_ids = self._get_tokenizer().encode(text)
        if len(token_ids) > self.MAX_PASSAGE_TOKENS:
            text = self._get_tokenizer().decode(token_ids[: self.MAX_PASSAGE_TOKENS])
        return text.strip()

    def download(self):
        os.makedirs(self.data_path, exist_ok=True)

        cached = self._cached_paths()
        if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in cached.values()):
            return

        import importlib
        hf_datasets = importlib.import_module("datasets")
        load_dataset = hf_datasets.load_dataset

        passage_rows = []
        query_rows = []
        qrel_rows = []
        papers_used = 0

        for split in self.SPLITS:
            dataset_split = load_dataset("allenai/qasper", revision="refs/convert/parquet", split=split)
            for paper in dataset_split:
                paper_id = paper["id"]
                full_text = paper["full_text"]
                paragraph_id_by_text = {}
                paper_passages = []
                for section_idx, section_paragraphs in enumerate(full_text["paragraphs"]):
                    for para_idx, paragraph_text in enumerate(section_paragraphs):
                        if not paragraph_text or not paragraph_text.strip():
                            continue
                        pid = f"{paper_id}_s{section_idx}_p{para_idx}"
                        paragraph_id_by_text.setdefault(paragraph_text, pid)
                        paper_passages.append((pid, self._truncate(paragraph_text)))

                qas = paper["qas"]
                paper_qrels = []
                paper_queries = []
                for i, question_id in enumerate(qas["question_id"]):
                    question_text = (qas["question"][i] or "").replace("\t", " ").strip()
                    if not question_text:
                        continue

                    answer_entries = qas["answers"][i]["answer"]
                    relevant_pids = set()
                    for answer in answer_entries:
                        if answer.get("unanswerable"):
                            continue
                        for evidence_text in answer.get("evidence") or []:
                            if "FLOAT SELECTED" in evidence_text:
                                continue
                            pid = paragraph_id_by_text.get(evidence_text)
                            if pid is not None:
                                relevant_pids.add(pid)

                    if not relevant_pids:
                        continue

                    paper_queries.append((question_id, question_text))
                    for pid in relevant_pids:
                        paper_qrels.append((question_id, pid))

                if not paper_queries:
                    continue

                papers_used += 1
                passage_rows.extend(paper_passages)
                query_rows.extend(paper_queries)
                qrel_rows.extend(paper_qrels)

        with open(cached["passages"], "w", encoding="utf-8") as f:
            for pid, text in passage_rows:
                f.write(f"{pid}\t{text}\n")

        with open(cached["queries"], "w", encoding="utf-8") as f:
            for qid, text in query_rows:
                f.write(f"{qid}\t{text}\n")

        with open(cached["answers"], "w", encoding="utf-8") as f:
            for qid, pid in qrel_rows:
                f.write(f"{qid}\t0\t{pid}\t1\n")

        print(f"Built Qasper cache: {papers_used} papers, {len(query_rows)} queries, {len(passage_rows)} passages, {len(qrel_rows)} qrels")

    def load_queries(self) -> dict[str, Query]:
        path = self._cached_paths()["queries"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Queries file not found at {path}. Run download() first.")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid, text = line.rstrip("\n").split("\t", 1)
                self.queries[qid] = Query(qid, text)
        return self.queries

    def load_passages(self) -> dict[str, Passage]:
        path = self._cached_paths()["passages"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Passages file not found at {path}. Run download() first.")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pid, text = line.rstrip("\n").split("\t", 1)
                self.passages[pid] = Passage(pid, text)
        return self.passages

    def load_answers(self) -> dict[str, Answer]:
        path = self._cached_paths()["answers"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Qrels file not found at {path}. Run download() first.")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                qid, _, pid, rel = line.rstrip("\n").split("\t")
                if int(rel) > 0:
                    if qid not in self.answers:
                        self.answers[qid] = Answer(relevant_passage_ids={pid})
                    else:
                        self.answers[qid].relevant_passage_ids.add(pid)
        return self.answers

    def load_all(self):
        self.download()
        self.load_queries()
        self.load_passages()
        self.load_answers()
