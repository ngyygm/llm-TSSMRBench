"""FAISS dense vector retriever with sentence-transformer embeddings."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .base import DATA_DIR, DatabaseEntry, Retriever, tokenize


def chunk_entries(
    entries: list[DatabaseEntry],
    chunk_size: int = 512,
    overlap: int = 64,
) -> tuple[list[str], list[dict]]:
    """Split entries into overlapping chunks for embedding-based retrieval."""
    chunk_texts: list[str] = []
    chunk_meta: list[dict] = []

    for e in entries:
        lines = e.text.split("\n", 1)
        prefix = lines[0]
        body = lines[1] if len(lines) > 1 else ""
        words = body.split()
        prefix_words = len(prefix.split())

        if len(words) + prefix_words <= chunk_size:
            chunk_texts.append(e.text)
            chunk_meta.append({
                "doc_id": e.doc_id,
                "source": e.source,
                "chunk_index": 0,
                "total_chunks": 1,
            })
        else:
            effective_chunk = chunk_size - prefix_words
            n_chunks = max(1, (len(words) - overlap) // max(1, effective_chunk - overlap))
            for i in range(n_chunks):
                start = i * max(1, effective_chunk - overlap)
                end = min(start + effective_chunk, len(words))
                chunk_texts.append(prefix + "\n" + " ".join(words[start:end]))
                chunk_meta.append({
                    "doc_id": e.doc_id,
                    "source": e.source,
                    "chunk_index": i,
                    "total_chunks": n_chunks,
                })
                if end >= len(words):
                    break

    return chunk_texts, chunk_meta


def _cache_path(model_name: str, chunk_size: int, overlap: int) -> Path:
    safe_model = model_name.replace("/", "_")
    return DATA_DIR / "cache" / f"{safe_model}_cs{chunk_size}_ov{overlap}.npz"


def _compute_chunk_hash(chunk_meta: list[dict]) -> str:
    h = hashlib.md5()
    for m in chunk_meta:
        h.update(f"{m['doc_id']}:{m['chunk_index']}".encode())
    return h.hexdigest()[:12]


def _encode_multi_gpu(
    model_name: str,
    chunk_texts: list[str],
    batch_size: int = 32,
) -> "np.ndarray":
    """Encode chunks across all available GPUs, then merge."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    n_gpus = torch.cuda.device_count()

    usable_gpus = []
    for i in range(n_gpus):
        try:
            free = torch.cuda.mem_get_info(i)[0]
            free_gb = free / (1024**3)
            if free_gb > 1.5:
                usable_gpus.append(i)
            else:
                print(f"    GPU {i}: {free_gb:.1f}GB free — skipping")
        except Exception:
            pass

    if not usable_gpus:
        print("  No GPU has enough memory, using CPU")
        model = SentenceTransformer(model_name, device="cpu")
        embs = model.encode(chunk_texts, batch_size=batch_size,
                            show_progress_bar=True, normalize_embeddings=True)
        return np.array(embs, dtype=np.float32)

    if len(usable_gpus) == 1:
        gpu_id = usable_gpus[0]
        print(f"  Using single GPU {gpu_id}")
        model = SentenceTransformer(model_name, device=f"cuda:{gpu_id}")
        embs = model.encode(chunk_texts, batch_size=batch_size,
                            show_progress_bar=True, normalize_embeddings=True)
        return np.array(embs, dtype=np.float32)

    chunks_per_gpu = len(chunk_texts) // len(usable_gpus)
    splits = []
    for idx, gpu_id in enumerate(usable_gpus):
        start = idx * chunks_per_gpu
        end = start + chunks_per_gpu if idx < len(usable_gpus) - 1 else len(chunk_texts)
        splits.append((start, end, gpu_id))

    print(f"  Multi-GPU encoding: {len(chunk_texts)} chunks across GPUs {usable_gpus}")
    for s, e, g in splits:
        print(f"    GPU {g}: chunks {s}-{e} ({e - s} chunks)")

    all_embs = []
    for start, end, gpu_id in splits:
        print(f"  Encoding on GPU {gpu_id}...")
        model = SentenceTransformer(model_name, device=f"cuda:{gpu_id}")
        batch = chunk_texts[start:end]
        embs = model.encode(batch, batch_size=batch_size,
                            show_progress_bar=True, normalize_embeddings=True)
        all_embs.append(np.array(embs, dtype=np.float32))
        del model
        torch.cuda.empty_cache()

    return np.concatenate(all_embs, axis=0)


class FAISSRetriever(Retriever):
    """Dense vector retrieval via sentence-transformers + FAISS."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        overlap: int = 64,
    ):
        self._model_name = model_name
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._index = None
        self._chunk_meta: list[dict] = []
        self._model = None

    def add_documents(self, docs: list[DatabaseEntry]) -> None:
        import json as _json
        import numpy as np
        import faiss as faiss_lib

        chunk_texts, self._chunk_meta = chunk_entries(docs, self._chunk_size, self._overlap)
        print(f"  Chunked {len(docs)} entries -> {len(chunk_texts)} chunks "
              f"(size={self._chunk_size}, overlap={self._overlap})")

        # Check cache
        cpath = _cache_path(self._model_name, self._chunk_size, self._overlap)
        chunk_hash = _compute_chunk_hash(self._chunk_meta)
        cache_meta_path = cpath.with_suffix(".meta.json")

        embeddings = None
        if cpath.exists() and cache_meta_path.exists():
            meta = _json.load(open(cache_meta_path))
            if meta.get("chunk_hash") == chunk_hash and meta.get("n_chunks") == len(chunk_texts):
                print(f"  Loading cached embeddings from {cpath} ...")
                data = np.load(cpath)
                embeddings = data["embeddings"]
                print(f"  Loaded: {embeddings.shape}")

        if embeddings is None:
            print(f"  No valid cache. Encoding with multi-GPU...")
            embeddings = _encode_multi_gpu(self._model_name, chunk_texts, batch_size=32)

            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cpath, embeddings=embeddings)
            _json.dump({
                "model": self._model_name,
                "chunk_size": self._chunk_size,
                "overlap": self._overlap,
                "n_chunks": len(chunk_texts),
                "chunk_hash": chunk_hash,
                "dim": embeddings.shape[1],
            }, open(cache_meta_path, "w"))
            print(f"  Cached embeddings to {cpath} ({embeddings.shape})")

        # Build FAISS index
        dim = embeddings.shape[1]
        self._index = faiss_lib.IndexFlatIP(dim)
        self._index.add(embeddings)
        print(f"  FAISS index built: {self._index.ntotal} vectors, dim={dim}")

        # Load model for query encoding
        import torch
        from sentence_transformers import SentenceTransformer

        query_device = "cpu"
        for i in range(torch.cuda.device_count()):
            try:
                free_gb = torch.cuda.mem_get_info(i)[0] / (1024**3)
                if free_gb > 1.5:
                    query_device = f"cuda:{i}"
                    break
            except Exception:
                pass
        print(f"  Loading model for queries: {self._model_name} on {query_device}")
        self._model = SentenceTransformer(self._model_name, device=query_device)

    def find(self, query: str, top_k: int = 10) -> list[int]:
        import numpy as np

        query_text = query
        if "e5" in self._model_name.lower():
            query_text = "query: " + query_text
        q_emb = self._model.encode([query_text], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)

        probe_k = min(max(top_k * 10, 100), self._index.ntotal)
        scores, indices = self._index.search(q_emb, probe_k)

        seen = set()
        doc_ids = []
        for idx in indices[0]:
            if idx < 0:
                continue
            doc_id = self._chunk_meta[idx]["doc_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
            if len(doc_ids) >= top_k:
                break
        return doc_ids

    def find_scored(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return (doc_id, score) pairs, deduplicated."""
        import numpy as np

        query_text = query
        if "e5" in self._model_name.lower():
            query_text = "query: " + query_text
        q_emb = self._model.encode([query_text], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)

        probe_k = min(max(top_k * 10, 100), self._index.ntotal)
        scores, indices = self._index.search(q_emb, probe_k)

        seen = set()
        results = []
        for i in range(probe_k):
            idx = indices[0][i]
            if idx < 0:
                continue
            doc_id = self._chunk_meta[idx]["doc_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                results.append((doc_id, float(scores[0][i])))
            if len(results) >= top_k:
                break
        return results
