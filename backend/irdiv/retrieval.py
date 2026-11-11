"""Information retrieval core: inverted index + BM25, TF-IDF space, LSA (dense) space."""
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class InvertedIndex:
    """term -> (doc ids, term frequencies); document frequency; BM25 scoring."""
    def __init__(self, docs):
        self.N = len(docs)
        self.dl = np.array([len(d["words"]) for d in docs], float)
        self.avgdl = self.dl.mean()
        post = defaultdict(dict)
        for d in docs:
            for t in d["words"]:
                post[t][d["id"]] = post[t].get(d["id"], 0) + 1
        self.post = {t: (np.fromiter(p.keys(), int), np.fromiter(p.values(), float)) for t, p in post.items()}
        self.df = {t: len(v[0]) for t, v in self.post.items()}

    def idf(self, t):
        f = self.df.get(t, 0)
        return np.log(1 + (self.N - f + .5) / (f + .5))

    def bm25(self, terms, k1=1.5, b=0.75):
        s = np.zeros(self.N)
        for t in terms:
            if t in self.post:
                ids, tf = self.post[t]
                s[ids] += self.idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.dl[ids] / self.avgdl))
        return s


class TfidfSpace:
    """L2-normalised TF-IDF vectors; cosine similarity between documents."""
    def __init__(self, docs):
        self.vec = TfidfVectorizer(sublinear_tf=True)
        self.X = self.vec.fit_transform([d["text"] for d in docs])

    def query_scores(self, text):
        return (self.X @ self.vec.transform([text]).T).toarray().ravel()

    def sim(self, ids):
        Xp = self.X[ids]
        return (Xp @ Xp.T).toarray()


class LsaSpace:
    """Dense proxy: truncated SVD (latent semantic analysis) of the TF-IDF matrix."""
    def __init__(self, tfidf, dim=32, seed=0):
        self.tf, self.svd = tfidf, TruncatedSVD(dim, random_state=seed)
        Z = self.svd.fit_transform(tfidf.X)
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    def query_scores(self, text):
        q = self.svd.transform(self.tf.vec.transform([text]))
        q = q / (np.linalg.norm(q) + 1e-12)
        return (self.Z @ q.T).ravel()

    def query_vec(self, text):
        q = self.svd.transform(self.tf.vec.transform([text]))
        return (q / (np.linalg.norm(q) + 1e-12)).ravel()
