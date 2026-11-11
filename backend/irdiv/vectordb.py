"""Minimal vector database: L2-normalised embeddings + metadata, cosine search.
Backend: FAISS (IndexFlatIP exact, or IndexHNSWFlat approximate) with a NumPy fallback.
For a larger corpus swap in Chroma / Qdrant / pgvector behind the same add() / search() interface."""
import json, numpy as np
try:
    import faiss
except ImportError:
    faiss = None


class VectorStore:
    def __init__(self, dim, kind="flat"):
        self.dim, self.meta = dim, []
        self.mat = np.zeros((0, dim), np.float32)          # kept for exact search and similarity matrices
        self.backend = "numpy"
        self.index = None
        if faiss is not None:
            self.index = (faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT) if kind == "hnsw"
                          else faiss.IndexFlatIP(dim))
            self.backend = f"faiss-{kind}"

    @staticmethod
    def _norm(v):
        v = np.asarray(v, np.float32).reshape(-1, v.shape[-1])
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    def add(self, vectors, meta):
        v = self._norm(np.asarray(vectors))
        if self.index is not None:
            self.index.add(v)
        self.mat = np.vstack([self.mat, v]); self.meta += list(meta)

    def search(self, q, k, exact_numpy=False):
        """Returns [(position, cosine score, metadata)] best first."""
        q = self._norm(np.asarray(q))
        k = min(k, len(self.meta))
        if self.index is not None and not exact_numpy:
            sc, ids = self.index.search(q, k); sc, ids = sc[0], ids[0]
        else:
            all_s = self.mat @ q[0]; ids = np.lexsort((np.arange(len(all_s)), -all_s))[:k]; sc = all_s[ids]
        return [(int(i), float(s), self.meta[i]) for i, s in zip(ids, sc) if i >= 0]

    def vectors(self, ids):
        return self.mat[list(ids)]

    def save(self, path):
        np.save(path + ".npy", self.mat); json.dump(self.meta, open(path + ".json", "w"))

    @classmethod
    def load(cls, path, kind="flat"):
        m = np.load(path + ".npy"); st = cls(m.shape[1], kind); st.add(m, json.load(open(path + ".json"))); return st
