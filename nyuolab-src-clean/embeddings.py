# NYU OLAB, 2023

# imports
import hashlib

import torch
from transformers import AutoModel, AutoTokenizer
from chromadb import Documents, Embeddings, EmbeddingFunction
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class HuggingFaceEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str, max_length: int = 64, device: str = None):
        """
        Base class for chromadb embedding functions that use HuggingFace models

        :param model_name: the name of the HuggingFace model to use
        :param max_length: the maximum length of the input text (default: 64)
        :param device: the device to use (default: None --> behavior: cpu, but use gpu if available)
        """
        super().__init__()

        self._setup_model_tokenizer(model_name=model_name)

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.torch_model.to(self.device)
        self.max_length = max_length

        print(f"[Embedding] Loaded {model_name} on device={self.device}", flush=True)

    def __call__(self, input: Documents) -> Embeddings:
        return self.get_embeddings(input)

    def _setup_model_tokenizer(self, model_name: str) -> None:
        """
        Sets up the model and tokenizer for the embedding function
        """
        self.torch_model = AutoModel.from_pretrained(model_name)
        self.torch_tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_embeddings(self, queries) -> list:
        """
        Consumes a list of queries
        Produces a list of embeddings

        :param queries: a list of queries
        :return: a list of embeddings
        """

        with torch.no_grad():
            encoded = self.torch_tokenizer(
                queries,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)

            embeds = self.torch_model(**encoded).last_hidden_state[:, 0, :].to("cpu")
            embeds = embeds.tolist()

            return embeds


class CachedEmbeddingFunction(EmbeddingFunction):
    """Transparent in-memory cache that wraps any EmbeddingFunction."""

    def __init__(self, base_ef: EmbeddingFunction):
        super().__init__()
        self._base = base_ef
        self._cache: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, input: Documents) -> Embeddings:
        results = [None] * len(input)
        to_compute_idx = []
        to_compute_txt = []

        for i, text in enumerate(input):
            key = hashlib.md5(text.encode("utf-8")).hexdigest()
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
                self.hits += 1
            else:
                to_compute_idx.append(i)
                to_compute_txt.append(text)
                self.misses += 1

        if to_compute_txt:
            new_embeds = self._base(to_compute_txt)
            for idx, text, emb in zip(to_compute_idx, to_compute_txt, new_embeds):
                key = hashlib.md5(text.encode("utf-8")).hexdigest()
                self._cache[key] = emb
                results[idx] = emb

        return results

    def report(self):
        total = self.hits + self.misses
        rate = self.hits / total * 100 if total > 0 else 0
        print(
            f"[EmbedCache] hits={self.hits} misses={self.misses} rate={rate:.1f}%",
            flush=True,
        )


class MedCPTEmbeddding(HuggingFaceEmbedding):
    """
    Class for the MedCPT-Query-Encoder embedding function for ChromaDB

    MedCPT Model Doc :: https://huggingface.co/ncbi/MedCPT-Query-Encoder
    """

    def __init__(self, **kwargs):
        super().__init__(model_name="ncbi/MedCPT-Query-Encoder", **kwargs)


class BertBaseEmbedding(HuggingFaceEmbedding):
    """
    Class for the BertBase embedding function for ChromaDB

    BertBase Model Doc :: https://huggingface.co/bert-base-uncased
    """

    def __init__(self, **kwargs):
        super().__init__(model_name="bert-base-uncased", **kwargs)


class ClinicalBertEmbedding(HuggingFaceEmbedding):
    """
    Class for the ClinicalBert embedding function for ChromaDB

    ClinicalBert Model Doc :: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
    """

    def __init__(self, **kwargs):
        super().__init__(model_name="emilyalsentzer/Bio_ClinicalBERT", **kwargs)


class _LazyEmbedDict(dict):
    """Only instantiate embedding models when first accessed."""

    _factories = {
        "medcpt": lambda: MedCPTEmbeddding(),
        "clinicalbert": lambda: ClinicalBertEmbedding(),
        "bertbase": lambda: BertBaseEmbedding(),
        "default": lambda: DefaultEmbeddingFunction(),
    }

    def __getitem__(self, key):
        if key not in dict.keys(self) and key in self._factories:
            print(f"[Embedding] Lazy-loading model: {key}", flush=True)
            dict.__setitem__(self, key, self._factories[key]())
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        return key in self._factories or dict.__contains__(self, key)


EMBED_MODEL_DICT = _LazyEmbedDict()
