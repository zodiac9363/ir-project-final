"""Vector DB + LLM stage.  Usage (from backend/): python run_vectordb.py --seeds 10 --out ../results
E11: does a vector-DB first stage and embedding-based coverage change the diversification result?
Also writes rag_demo.json (seed-0 trace with vector-DB hits and LLM explanations) for the front end."""
import argparse, json, numpy as np, pandas as pd
from run_experiments import World, ranks, pool_data, first_stage, K, P, LAM
from irdiv.generator import QUERIES, query_terms, ISSUES
from irdiv.embeddings import get_embedder
from irdiv.vectordb import VectorStore
from irdiv.diversify import greedy, mmr
from irdiv.metrics import Evaluator
from irdiv.llm import get_llm, explain_selection, rewrite_case

nm = lambda w, i: [ISSUES[x][0] for x in w.docs[i]["issues"]]


def build_store(w, kind="auto"):
    emb = get_embedder(w.docs, kind); Z = emb.encode_docs()
    st = VectorStore(Z.shape[1]); st.add(Z, [dict(id=d["id"], issues=list(d["issues"])) for d in w.docs])
    return emb, st


def hybrid_pool(w, emb, st, q, pool_size):
    """BM25 ranking + vector-DB ranking fused with reciprocal rank fusion."""
    qt = query_terms(q); b = w.index.bm25(qt); d = np.zeros(w.N)
    for j, s, _ in st.search(emb.encode_query(" ".join(qt)), w.N):
        d[j] = s
    f = 1 / (60 + ranks(b)) + 1 / (60 + ranks(d))
    pool = np.lexsort((np.arange(w.N), -f))[:pool_size]
    return pool, f[pool] / f[pool].max()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--out", default="../results")
    a = ap.parse_args(); rows = []
    for s in range(a.seeds):
        w = World(s, .3); emb, st = build_store(w)
        for q in QUERIES:
            if q["control"]: continue
            ev = Evaluator(w.docs, q["issues"], K)
            bp, br, _ = (lambda t: (t[0], t[1], t[2]))(pool_data(w, first_stage(w, q, "BM25"), P))
            hp, hr = hybrid_pool(w, emb, st, q, P)
            for fs, pool, r in (("BM25 pool", bp, br), ("Vector DB + BM25 pool", hp, hr)):
                for sim in ("TF-IDF cosine", "Vector-DB embeddings"):
                    S = w.tfidf.sim(pool) if sim.startswith("TF") else np.maximum(st.vectors(pool) @ st.vectors(pool).T, 0)
                    for rr, sel in (("top-K", list(range(K))), ("MMR", mmr(r, S, K, LAM)), ("Submodular", greedy(r, S, K, LAM)[0])):
                        rows.append(dict(first_stage=fs, similarity=sim, rerank=rr, **ev.evaluate(pool[sel])))
    df = pd.DataFrame(rows).groupby(["first_stage", "similarity", "rerank"])[["P", "S_recall", "alpha_NDCG", "Dups"]].mean().round(3).reset_index()
    df.to_csv(f"{a.out}/E11_vectordb_grid.csv", index=False); print(df.to_string(index=False))

    # demo trace + FAISS vs exact NumPy parity
    w = World(0, .3); emb, st = build_store(w); llm = get_llm(); out = []; same = []
    for q in QUERIES:
        qv = emb.encode_query(" ".join(query_terms(q)))
        same.append({h[0] for h in st.search(qv, 10)} == {h[0] for h in st.search(qv, 10, exact_numpy=True)})
        pool, r = hybrid_pool(w, emb, st, q, P); S = np.maximum(st.vectors(pool) @ st.vectors(pool).T, 0)
        sel, g = greedy(r, S, K, LAM); bm = w.index.bm25(query_terms(q)); rk = {int(i): j + 1 for j, i in enumerate(np.lexsort((np.arange(w.N), -bm)))}
        picks = [dict(id=int(pool[c]), issues=nm(w, pool[c]), dup=bool(w.docs[pool[c]]["dup"]), root=int(w.docs[pool[c]]["root"]),
                      rank=rk[int(pool[c])], gain=round(x, 4)) for c, x in zip(sel, g)]
        qi = [ISSUES[i][0] for i in q["issues"]]
        out.append(dict(name=q["name"], dense_top=[dict(id=j, score=round(s, 3), issues=nm(w, j)) for j, s, _ in st.search(qv, 5)],
                        picks=picks, explanation=explain_selection(llm, q["name"], qi, picks),
                        rewrites=[rewrite_case(llm, w.docs[p["id"]]["words"], p["issues"]) for p in picks[:2]] if not llm.offline else None))
    meta = dict(embedder=emb.name, vectordb=st.backend, dim=int(st.dim), n_vectors=len(st.meta), llm=llm.name,
                faiss_matches_exact_numpy=bool(all(same)), llm_is_real=not llm.offline)
    json.dump(dict(meta=meta, queries=out), open(f"{a.out}/rag_demo.json", "w"), indent=1); print(meta)


if __name__ == "__main__":
    main()
