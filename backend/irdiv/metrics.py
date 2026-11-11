"""Relevance (P, Recall, AP, NDCG, MRR) and diversity (S-recall, alpha-NDCG) metrics.
A case is relevant if it covers at least one of the query's legal issues."""
from collections import Counter
import numpy as np


class Evaluator:
    def __init__(self, docs, q_issues, k, flip=0.0, rng=None, alpha=0.5):
        self.docs, self.qi, self.k, self.alpha = docs, set(q_issues), k, alpha
        true = np.array([bool(self.qi & set(d["issues"])) for d in docs])
        self.rel = true ^ (rng.random(len(docs)) < flip) if flip > 0 else true   # optional label noise
        self.R = max(int(self.rel.sum()), 1)
        self.ideal = self._ideal(np.flatnonzero(true))
        self.ia = max(self._dcg(self.gains(self.ideal)), 1e-9)
        self.ii = max(self._dcg(np.ones(min(self.R, k))), 1e-9)

    @staticmethod
    def _dcg(g):
        g = np.asarray(g, float)
        return float((g / np.log2(np.arange(2, len(g) + 2))).sum())

    def _gain(self, i, seen):
        return sum(self.alpha ** seen[x] for x in self.docs[i]["issues"] if x in self.qi)

    def gains(self, ids):
        seen, out = Counter(), []
        for i in ids:
            out.append(self._gain(i, seen))
            for x in self.docs[i]["issues"]:
                if x in self.qi:
                    seen[x] += 1
        return out

    def _ideal(self, cand):
        """Greedy ideal ranking for alpha-NDCG (also used as the label-cheating oracle)."""
        cand, seen, chosen = list(cand), Counter(), []
        for _ in range(self.k):
            if not cand:
                break
            i = max(cand, key=lambda c: self._gain(c, seen))
            cand.remove(i); chosen.append(int(i))
            for x in self.docs[i]["issues"]:
                if x in self.qi:
                    seen[x] += 1
        return chosen

    def evaluate(self, ids):
        ids = [int(i) for i in ids][: self.k]
        r = self.rel[ids].astype(float)
        hits = np.cumsum(r)
        ap = float((r * hits / np.arange(1, len(r) + 1)).sum() / min(self.R, self.k))
        first = np.flatnonzero(r)
        cov = set()
        for i in ids:
            if self.rel[i]:
                cov |= self.qi & set(self.docs[i]["issues"])
        dups = sum(v - 1 for v in Counter(self.docs[i]["root"] for i in ids).values())
        return dict(P=r.mean(), Recall=r.sum() / self.R, AP=ap, NDCG=self._dcg(r) / self.ii,
                    MRR=1 / (first[0] + 1) if len(first) else 0.0, S_recall=len(cov) / len(self.qi),
                    alpha_NDCG=self._dcg(self.gains(ids)) / self.ia, Dups=dups)
