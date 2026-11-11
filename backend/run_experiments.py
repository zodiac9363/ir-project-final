"""Runs experiments E1-E10 and writes CSV tables + figures to ./results.
Usage:  python run_experiments.py --seeds 10 --out results"""
import argparse, os, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from irdiv.generator import make_corpus, QUERIES, query_terms
from irdiv.retrieval import InvertedIndex, TfidfSpace, LsaSpace
from irdiv.diversify import greedy, lazy_greedy, exact, mmr, objective
from irdiv.metrics import Evaluator

K, P, LAM = 10, 100, 0.7
METRICS = ["P", "Recall", "AP", "NDCG", "MRR", "S_recall", "alpha_NDCG", "Dups"]


class World:
    def __init__(self, seed, dup, n=400):
        self.docs = make_corpus(n, dup, seed); self.N = n
        self.index, self.tfidf, self._lsa = InvertedIndex(self.docs), TfidfSpace(self.docs), None

    @property
    def lsa(self):
        if self._lsa is None:
            self._lsa = LsaSpace(self.tfidf)
        return self._lsa


def ranks(s):
    o = np.lexsort((np.arange(len(s)), -s)); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); return r


def first_stage(w, q, kind):
    qt = query_terms(q); b = w.index.bm25(qt)
    if kind == "BM25":
        return b
    d = w.lsa.query_scores(" ".join(qt))
    return d if kind == "Dense" else 1 / (60 + ranks(b)) + 1 / (60 + ranks(d))   # reciprocal rank fusion


def pool_data(w, scores, pool_size):
    pool = np.lexsort((np.arange(len(scores)), -scores))[:pool_size]
    s = scores[pool]
    r = s / (s.max() or 1) if s.min() >= 0 else (s - s.min()) / (s.max() - s.min() + 1e-12)
    return pool, r, w.tfidf.sim(pool)


def run_systems(w, q, ev, rng, k=K, pool_size=P, lam=LAM, weighted=True):
    qt = query_terms(q)
    pool, r, S = pool_data(w, first_stage(w, q, "BM25"), pool_size)
    tf = w.tfidf.query_scores(" ".join(qt))
    out = {"TF-IDF": np.lexsort((np.arange(w.N), -tf))[:k], "BM25": pool[:k],
           "MMR": pool[mmr(r, S, k, lam)], "Submodular": pool[greedy(r, S, k, lam, weighted)[0]],
           "Random": rng.choice(pool, k, replace=False), "Oracle": np.array(ev.ideal)}
    return out, (pool, r, S)


def row(**kw):
    return kw


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--out", default="results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    save = lambda df, name: (df.to_csv(f"{a.out}/{name}.csv", index=False), print(f"\n== {name} ==\n{df.round(3).to_string(index=False)}"))
    seeds, DUPS = range(a.seeds), [0.0, 0.3, 0.6]

    # ---------- E1/E2 systems x data conditions, plus per-run features for E8/E9 ----------
    rows, feats = [], []
    for dup in DUPS:
        for s in seeds:
            w = World(s, dup); rng = np.random.default_rng(s)
            for qi, q in enumerate(QUERIES):
                ev = Evaluator(w.docs, q["issues"], K); out, (pool, r, S) = run_systems(w, q, ev, rng)
                res = {n: ev.evaluate(ids) for n, ids in out.items()}
                for n, m in res.items():
                    rows.append(row(dup=dup, seed=s, query=q["name"], control=q["control"], system=n, **m))
                prim = np.bincount([w.docs[i]["issues"][0] for i in pool], minlength=6)
                feats.append(dict(dup=dup, seed=s, query=q["name"], control=q["control"], n_issues=len(q["issues"]),
                                  q_len=len(query_terms(q)), top_issue_share=prim.max() / len(pool),
                                  pool_mean_sim=(S.sum() - len(S)) / (len(S) ** 2 - len(S)),
                                  bm25_dups=res["BM25"]["Dups"], score_gap=r[0] - r[K - 1],
                                  delta_alpha=res["Submodular"]["alpha_NDCG"] - res["BM25"]["alpha_NDCG"]))
    main_df = pd.DataFrame(rows); fe = pd.DataFrame(feats)
    order = ["TF-IDF", "BM25", "MMR", "Random", "Submodular", "Oracle"]
    save(main_df.groupby(["control", "system"])[METRICS].mean().reindex(order, level=1).reset_index(), "E1_systems_by_query_type")
    save(main_df[main_df.system.isin(["BM25", "MMR", "Submodular"])].groupby(["dup", "system"])[METRICS].mean().reset_index(), "E2_by_duplicate_rate")

    # ---------- E3 lambda sweep ----------
    rows = []
    for s in seeds:
        w = World(s, .3); rng = np.random.default_rng(s)
        for q in QUERIES:
            if q["control"]: continue
            ev = Evaluator(w.docs, q["issues"], K); pool, r, S = pool_data(w, first_stage(w, q, "BM25"), P)
            for lam in np.round(np.arange(0, 1.01, .1), 1):
                for n, sel in (("Submodular", greedy(r, S, K, lam)[0]), ("MMR", mmr(r, S, K, lam))):
                    rows.append(row(lam=lam, system=n, **ev.evaluate(pool[sel])))
    e3 = pd.DataFrame(rows).groupby(["system", "lam"])[METRICS].mean().reset_index(); save(e3, "E3_lambda_sweep")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for m, c in (("S_recall", "#1f3a5f"), ("alpha_NDCG", "#8a6d3b"), ("P", "#3f6b5a"), ("Dups", "#7a2e3a")):
        d = e3[e3.system == "Submodular"]; ax.plot(d.lam, d[m] / (10 if m == "Dups" else 1), label=m + (" /10" if m == "Dups" else ""), color=c)
    ax.set_xlabel("lambda (0 = BM25, 1 = pure variety)"); ax.legend(); ax.set_title("E3: relevance vs variety (multi-issue queries)")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_E3_lambda.png", dpi=150); plt.close(fig)

    # ---------- E4 pool size x k ; E5 ablations ; E7 label noise ----------
    e4, e5, e7 = [], [], []
    for s in seeds:
        w = World(s, .3); rng = np.random.default_rng(s)
        for q in QUERIES:
            if q["control"]: continue
            for k in (5, 10, 15):
                ev = Evaluator(w.docs, q["issues"], k)
                for ps in (20, 50, 100, 200):
                    out, _ = run_systems(w, q, ev, rng, k=k, pool_size=ps)
                    for n in ("BM25", "Submodular"):
                        e4.append(row(k=k, pool=ps, system=n, **ev.evaluate(out[n])))
            ev = Evaluator(w.docs, q["issues"], K)
            for name, lam, wt in (("full (lam=.7)", .7, True), ("relevance only (lam=0)", 0, True),
                                  ("coverage only (lam=1)", 1, True), ("unweighted coverage", .7, False)):
                out, _ = run_systems(w, q, ev, rng, lam=lam, weighted=wt); e5.append(row(variant=name, **ev.evaluate(out["Submodular"])))
            for flip in (0, .05, .1, .2):
                evn = Evaluator(w.docs, q["issues"], K, flip=flip, rng=np.random.default_rng(s))
                out, _ = run_systems(w, q, evn, rng)
                for n in ("BM25", "Submodular"):
                    e7.append(row(label_noise=flip, system=n, **evn.evaluate(out[n])))
    save(pd.DataFrame(e4).groupby(["k", "pool", "system"])[["S_recall", "alpha_NDCG", "P"]].mean().reset_index(), "E4_pool_and_k")
    save(pd.DataFrame(e5).groupby("variant")[METRICS].mean().reset_index(), "E5_ablation")
    save(pd.DataFrame(e7).groupby(["label_noise", "system"])[["P", "NDCG", "S_recall"]].mean().reset_index(), "E7_label_noise")

    # ---------- E6 greedy vs exact optimum, lazy greedy savings ----------
    rows = []
    for s in range(min(a.seeds, 5)):
        w = World(s, .3)
        for q in QUERIES:
            for lam in (.3, .7):
                pool, r, S = pool_data(w, first_stage(w, q, "BM25"), 20)
                g = objective(greedy(r, S, 4, lam)[0], r, S, 4, lam); o = objective(exact(r, S, 4, lam), r, S, 4, lam)
                pool, r, S = pool_data(w, first_stage(w, q, "BM25"), P)
                sel, ev_n = lazy_greedy(r, S, K, lam)
                rows.append(row(query=q["name"], lam=lam, greedy_over_opt=g / o, lazy_evals=ev_n, naive_evals=K * P,
                                same_set=set(sel) == set(greedy(r, S, K, lam)[0])))
    e6 = pd.DataFrame(rows); save(e6.groupby("lam").agg(min_ratio=("greedy_over_opt", "min"), mean_ratio=("greedy_over_opt", "mean"),
                                                         lazy_evals=("lazy_evals", "mean"), naive_evals=("naive_evals", "mean"),
                                                         same_set=("same_set", "mean")).reset_index(), "E6_greedy_vs_optimum")

    # ---------- E8 helps / neutral / hurts ----------
    fe["verdict"] = np.where(fe.delta_alpha > .03, "helps", np.where(fe.delta_alpha < -.03, "hurts", "neutral"))
    save(fe.groupby(["control", "verdict"]).size().unstack(fill_value=0).reset_index(), "E8_helps_neutral_hurts")

    # ---------- E9 predict benefit from per-query features (cross-validated) ----------
    cols = ["n_issues", "q_len", "top_issue_share", "pool_mean_sim", "bm25_dups", "score_gap"]
    mdl = make_pipeline(StandardScaler(), Ridge(1.0)); sc = cross_val_score(mdl, fe[cols], fe.delta_alpha, cv=5, scoring="r2")
    mdl.fit(fe[cols], fe.delta_alpha)
    save(pd.DataFrame({"feature": cols, "std_coef": mdl[-1].coef_}).assign(cv_r2_mean=sc.mean()), "E9_regression")

    # ---------- E10 first stage x re-ranker ----------
    rows = []
    for s in seeds:
        w = World(s, .3); rng = np.random.default_rng(s)
        for q in QUERIES:
            if q["control"]: continue
            ev = Evaluator(w.docs, q["issues"], K)
            for fs in ("BM25", "Dense", "Hybrid"):
                pool, r, S = pool_data(w, first_stage(w, q, fs), P)
                for rr, sel in (("top-K", list(range(K))), ("MMR", mmr(r, S, K, LAM)), ("Submodular", greedy(r, S, K, LAM)[0])):
                    rows.append(row(first_stage=fs, rerank=rr, **ev.evaluate(pool[sel])))
    save(pd.DataFrame(rows).groupby(["first_stage", "rerank"])[["P", "NDCG", "S_recall", "alpha_NDCG", "Dups"]].mean().reset_index(), "E10_grid")

    # ---------- data sanity check: does TF-IDF + k-means recover the issues? ----------
    w = World(0, 0.0); lab = [d["issues"][0] for d in w.docs]
    km = KMeans(6, n_init=10, random_state=0).fit(w.tfidf.X)
    save(pd.DataFrame([dict(check="k-means(6) vs main issue, adjusted Rand", value=adjusted_rand_score(lab, km.labels_))]), "data_check_kmeans")

    # ---------- figure: systems ----------
    d = main_df[~main_df.control].groupby("system")[["S_recall", "alpha_NDCG", "P"]].mean().reindex(order)
    ax = d.plot.bar(figsize=(7, 3.8), color=["#1f3a5f", "#8a6d3b", "#3f6b5a"], rot=20)
    ax.set_title("E1: systems on multi-issue queries"); plt.tight_layout(); plt.savefig(f"{a.out}/fig_E1_systems.png", dpi=150)
    print(f"\nDone in {time.time()-t0:.0f}s. Files in {a.out}/")


if __name__ == "__main__":
    main()
