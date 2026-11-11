"""Extra experiments.  Usage (from backend/): python run_extras.py --seeds 10 --out ../results
E12 lambda calibration + learned lambda | E13 speed | E14 baselines (xQuAD, learning-to-rank) with bootstrap intervals"""
import argparse, time, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from run_experiments import World, pool_data, first_stage, K, P, LAM
from run_vectordb import build_store, hybrid_pool
from irdiv.generator import QUERIES, query_terms
from irdiv.diversify import greedy, lazy_greedy, mmr
from irdiv.metrics import Evaluator

GRID = [.3, .5, .7, .9, 1.0]


def xquad(r, S, Xp, k, lam, n_aspects=4):
    lab = KMeans(n_aspects, n_init=5, random_state=0).fit_predict(Xp)
    pa = np.array([r[lab == a].sum() for a in range(n_aspects)]); pa = pa / pa.sum()
    cov, sel = np.ones(n_aspects), []
    for _ in range(k):
        sc = []
        for d in range(len(r)):
            if d in sel: sc.append(-1); continue
            sc.append((1 - lam) * r[d] + lam * sum(pa[a] * (r[d] if lab[d] == a else 0) * cov[a] for a in range(n_aspects)))
        c = int(np.argmax(sc)); sel.append(c); cov[lab[c]] *= (1 - r[c])
    return sel


def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x); m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--out", default="../results"); a = ap.parse_args()
    save = lambda df, n: (df.round(3).to_csv(f"{a.out}/{n}.csv", index=False), print(f"\n== {n} ==\n{df.round(3).to_string(index=False)}"))
    rows, base, ltr_rows = [], [], []
    for s in range(a.seeds):
        w = World(s, .3); emb, st = build_store(w)
        for q in QUERIES:
            if q["control"]: continue
            ev = Evaluator(w.docs, q["issues"], K); qt = " ".join(query_terms(q))
            dense = np.zeros(w.N)
            for j, sc_, _ in st.search(emb.encode_query(qt), w.N): dense[j] = sc_
            tfs = w.tfidf.query_scores(qt)
            for name in ("BM25 pool", "Hybrid pool"):
                pool, r = (pool_data(w, first_stage(w, q, "BM25"), P)[:2] if name == "BM25 pool" else hybrid_pool(w, emb, st, q, P))
                S = w.tfidf.sim(pool); res = {}
                for lam in GRID:
                    res[lam] = ev.evaluate(pool[greedy(r, S, K, lam)[0]])
                    rows.append(dict(seed=s, pool=name, lam=lam, **{m: res[lam][m] for m in ("S_recall", "alpha_NDCG", "P")}))
                best = max(GRID, key=lambda l: res[l]["alpha_NDCG"])
                feats = dict(seed=s, pool=name, best_lam=best, r_std=r.std(), r_mean=r.mean(), r_gap=r[0] - r[K - 1], mean_sim=(S.sum() - len(S)) / (len(S) ** 2 - len(S)),
                             n_issues=len(q["issues"]), alpha_by_lam={l: res[l]["alpha_NDCG"] for l in GRID})
                ltr_rows.append(feats)
                if name == "BM25 pool":
                    sel = pool[greedy(r, S, K, LAM)[0]]
                    Xp = w.tfidf.X[pool].toarray()
                    base.append(dict(seed=s, q=q["name"], **{f"{sy}|{m}": ev.evaluate(ids)[m] for sy, ids in (
                        ("BM25", pool[:K]), ("MMR", pool[mmr(r, S, K, LAM)]), ("xQuAD (k-means aspects)", pool[xquad(r, S, Xp, K, LAM)]), ("Submodular", sel)) for m in ("S_recall", "alpha_NDCG", "P")}))
                    # pointwise learning-to-rank on [bm25 score, tfidf cos, dense cos, length]; trained on other seeds
                    F = np.c_[r, tfs[pool], dense[pool], [len(w.docs[i]["words"]) for i in pool]]
                    y = [ev.rel[i] * 1.0 for i in pool]
                    ltr_rows[-1]["_ltr"] = (F, y, pool, ev)
    # ---- E12: lambda calibration and learned lambda
    d = pd.DataFrame(rows); save(d.groupby(["pool", "lam"])[["S_recall", "alpha_NDCG", "P"]].mean().reset_index(), "E12a_lambda_by_pool")
    L = pd.DataFrame([{k: v for k, v in r.items() if k not in ("alpha_by_lam", "_ltr")} for r in ltr_rows]); cols = ["r_std", "r_mean", "r_gap", "mean_sim", "n_issues"]
    mdl = make_pipeline(StandardScaler(), Ridge(1.0)); pred = cross_val_predict(mdl, L[cols], L.best_lam, cv=GroupKFold(min(5, a.seeds)), groups=L.seed)
    snap = lambda x: min(GRID, key=lambda g: abs(g - x))
    ab = [r["alpha_by_lam"] for r in ltr_rows]
    out = pd.DataFrame([dict(strategy=n, alpha_NDCG=np.mean([ab[i][f(i)] for i in range(len(ab))])) for n, f in
                        (("fixed lambda 0.7", lambda i: .7), ("fixed lambda 0.9", lambda i: .9), ("learned lambda (cross-validated ridge)", lambda i: snap(pred[i])),
                         ("best lambda per query (oracle)", lambda i: L.best_lam[i]))])
    save(out, "E12b_learned_lambda")
    # ---- E13: speed
    w = World(0, .3); q = QUERIES[0]; rows13 = []
    for ps in (50, 100, 200, 400):
        pool, r, S = pool_data(w, first_stage(w, q, "BM25"), ps)
        for n, f in (("greedy", lambda: greedy(r, S, K, LAM)), ("lazy greedy", lambda: lazy_greedy(r, S, K, LAM)), ("MMR", lambda: mmr(r, S, K, LAM))):
            t = time.perf_counter(); [f() for _ in range(20)]; rows13.append(dict(pool=ps, method=n, ms=(time.perf_counter() - t) / 20 * 1000))
        rows13.append(dict(pool=ps, method="lazy gain evaluations", ms=float(lazy_greedy(r, S, K, LAM)[1])))
    save(pd.DataFrame(rows13), "E13_speed")
    # ---- E14: baselines + bootstrap intervals; learning to rank with cross-validation over seeds
    B = pd.DataFrame(base); sy = ["BM25", "MMR", "xQuAD (k-means aspects)", "Submodular"]
    # LTR
    ltr = [r for r in ltr_rows if r["pool"] == "BM25 pool"]; seeds = np.array([r["seed"] for r in ltr]); lt = []
    for tr_s in range(a.seeds):
        tr = [r["_ltr"] for r, s_ in zip(ltr, seeds) if s_ != tr_s]; te = [r for r, s_ in zip(ltr, seeds) if s_ == tr_s]
        m = Ridge(1.0).fit(np.vstack([t[0] for t in tr]), np.concatenate([t[1] for t in tr]))
        for r in te:
            Fq, yq, pool, ev = r["_ltr"]; order = pool[np.argsort(-m.predict(Fq), kind="stable")[:K]]; e = ev.evaluate(order)
            lt.append(dict(seed=tr_s, **{f"LTR (ridge)|{k_}": e[k_] for k_ in ("S_recall", "alpha_NDCG", "P")}))
    LT = pd.DataFrame(lt); tab = []
    for name in sy:
        tab.append(dict(system=name, **{m: B[f"{name}|{m}"].mean() for m in ("P", "S_recall", "alpha_NDCG")}))
    tab.append(dict(system="LTR (ridge)", **{m: LT[f"LTR (ridge)|{m}"].mean() for m in ("P", "S_recall", "alpha_NDCG")}))
    save(pd.DataFrame(tab), "E14a_baselines")
    ci = []
    for other in ("BM25", "MMR", "xQuAD (k-means aspects)"):
        for m in ("alpha_NDCG", "S_recall"):
            mn, lo, hi = boot(B[f"Submodular|{m}"] - B[f"{other}|{m}"]); ci.append(dict(comparison=f"Submodular minus {other}", metric=m, mean_diff=mn, ci_low=lo, ci_high=hi, runs=len(B)))
    save(pd.DataFrame(ci), "E14b_bootstrap_intervals")


if __name__ == "__main__":
    main()
