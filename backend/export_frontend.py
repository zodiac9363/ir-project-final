"""Exports a pipeline trace (seed 0) + all experiment tables to frontend_data.json for the web page.
Usage (from backend/): python run_experiments.py --out ../results && python export_frontend.py"""
import json, numpy as np, pandas as pd
from run_experiments import World, run_systems, K, P, LAM
from irdiv.generator import QUERIES, query_terms, ISSUES
from irdiv.metrics import Evaluator
from irdiv.diversify import greedy, mmr


def case(w, i, **kw):
    d = w.docs[i]
    return dict(id=int(i), issues=[ISSUES[x][0] for x in d["issues"]], dup=bool(d["dup"]), root=int(d["root"]), **kw)


def build(seed=0, dup=0.3, res="results"):
    w = World(seed, dup); rng = np.random.default_rng(seed); qs = []
    for q in QUERIES:
        ev = Evaluator(w.docs, q["issues"], K); out, (pool, r, S) = run_systems(w, q, ev, rng)
        terms = query_terms(q); sc = w.index.bm25(terms); rank = {int(p): j + 1 for j, p in enumerate(pool)}
        sel, gains = greedy(r, S, K, LAM); ms = mmr(r, S, K, LAM)
        prim = np.bincount([w.docs[i]["issues"][0] for i in pool], minlength=6)
        qs.append(dict(
            name=q["name"], control=q["control"], issues=[ISSUES[i][0] for i in q["issues"]], terms=terms,
            idf=[dict(term=t, df=w.index.df.get(t, 0), idf=round(float(w.index.idf(t)), 3)) for t in dict.fromkeys(terms)],
            pool_issues={ISSUES[i][0]: int(c) for i, c in enumerate(prim)},
            bm25=[case(w, int(i), score=round(float(sc[i]), 2), rank=j + 1) for j, i in enumerate(pool[:K])],
            sub=[case(w, int(pool[c]), score=round(float(sc[pool[c]]), 2), rank=rank[int(pool[c])], gain=round(g, 4)) for c, g in zip(sel, gains)],
            mmr=[case(w, int(pool[c]), score=round(float(sc[pool[c]]), 2), rank=rank[int(pool[c])]) for c in ms],
            metrics={n: {k: round(float(v), 3) for k, v in ev.evaluate(ids).items()} for n, ids in out.items()}))
    tabs = {n: pd.read_csv(f"{res}/{f}.csv").round(3).to_dict("records") for n, f in [
        ("E1", "E1_systems_by_query_type"), ("E2", "E2_by_duplicate_rate"), ("E3", "E3_lambda_sweep"), ("E4", "E4_pool_and_k"),
        ("E5", "E5_ablation"), ("E6", "E6_greedy_vs_optimum"), ("E7", "E7_label_noise"), ("E8", "E8_helps_neutral_hurts"),
        ("E9", "E9_regression"), ("E10", "E10_grid"), ("E11", "E11_vectordb_grid"), ("E12a", "E12a_lambda_by_pool"), ("E12b", "E12b_learned_lambda"), ("E13", "E13_speed"), ("E14a", "E14a_baselines"), ("E14b", "E14b_bootstrap_intervals"), ("kmeans", "data_check_kmeans")]}
    import os
    for key, f in (("LAB", "label_agreement"), ("LLMJ", "llm_judge"), ("ILDC", "ildc_expert_eval")):
        if os.path.exists(f"{res}/{f}.csv"):
            tabs[key] = pd.read_csv(f"{res}/{f}.csv").round(3).to_dict("records")
    meta = dict(seed=seed, dup=dup, k=K, pool=P, lam=LAM, N=w.N, V=len(w.index.post), avgdl=round(float(w.index.avgdl), 1),
                n_dup=sum(d["dup"] for d in w.docs))
    return dict(meta=meta, queries=qs, tables=tabs, rag=json.load(open(f"{res}/rag_demo.json")))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="../results"); ap.add_argument("--out", default="../data/frontend_data.json")
    a = ap.parse_args()
    json.dump(build(res=a.res), open(a.out, "w"), separators=(",", ":"))
    print("wrote", a.out)
