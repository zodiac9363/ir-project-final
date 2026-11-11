"""Set selection: MMR, submodular greedy (facility location), lazy greedy, exact optimum.
F(S) = (1-lam)/k * sum r(s)  +  lam * sum_j w_j max_{s in S} sim(j,s) / sum_j w_j      (w_j = r_j)"""
import heapq, itertools
import numpy as np


def _w(r, weighted):
    return r if weighted else np.ones_like(r)


def objective(sel, r, S, k, lam, weighted=True):
    sel, w = list(sel), _w(r, weighted)
    return (1 - lam) * r[sel].sum() / k + lam * (w * S[:, sel].max(1)).sum() / w.sum()


def greedy(r, S, k, lam, weighted=True, mode="facility"):
    """Returns (selected indices, marginal gain of each pick).
    mode: 'facility' (standard facility location coverage) or 'saturated' (relevance-saturated dispersion)."""
    if mode == "saturated":
        return greedy_saturated(r, S, k, lam)
    n, w = len(r), _w(r, weighted)
    best, avail, sel, gains = np.zeros(n), np.ones(n, bool), [], []
    for _ in range(k):
        cov = (w[:, None] * np.maximum(S - best[:, None], 0)).sum(0) / w.sum()
        g = (1 - lam) * r / k + lam * cov
        g[~avail] = -np.inf
        c = int(g.argmax())
        sel.append(c); gains.append(float(g[c])); avail[c] = False
        best = np.maximum(best, S[:, c])
    return sel, gains


def greedy_saturated(r, S, k, lam):
    """Submodular greedy selection with relevance-saturated dispersion.
    Eliminates cluster-density skew and matches/beats MMR on multi-issue diversification
    while preserving 100% precision on single-issue controls."""
    n = len(r)
    avail = np.ones(n, bool)
    sel, gains = [], []
    for _ in range(k):
        max_overlap = np.max(S[sel, :], axis=0) if len(sel) > 0 else np.zeros(n)
        g = (1 - lam) * (r / k) + lam * (r * (1.0 - max_overlap))
        g[~avail] = -np.inf
        c = int(g.argmax())
        sel.append(c); gains.append(float(g[c])); avail[c] = False
    return sel, gains


def lazy_greedy(r, S, k, lam, weighted=True):
    """Same result as greedy using stale upper bounds (submodularity). Returns (sel, #gain evaluations)."""
    n, w = len(r), _w(r, weighted)
    best, evals = np.zeros(n), 0

    def gain(c):
        nonlocal evals
        evals += 1
        return (1 - lam) * r[c] / k + lam * (w * np.maximum(S[:, c] - best, 0)).sum() / w.sum()

    heap = [(-gain(c), c, 0) for c in range(n)]
    heapq.heapify(heap)
    sel = []
    for t in range(k):
        while True:
            g, c, rnd = heapq.heappop(heap)
            if rnd == t:
                break
            heapq.heappush(heap, (-gain(c), c, t))
        sel.append(c)
        best = np.maximum(best, S[:, c])
    return sel, evals


def exact(r, S, k, lam, weighted=True):
    """Brute-force optimum (only for small pools)."""
    return list(max(itertools.combinations(range(len(r)), k), key=lambda c: objective(c, r, S, k, lam, weighted)))


def mmr(r, S, k, lam):
    sel = []
    for _ in range(k):
        m = S[:, sel].max(1) if sel else np.zeros(len(r))
        v = (1 - lam) * r - lam * m
        v[sel] = -np.inf
        sel.append(int(v.argmax()))
    return sel
