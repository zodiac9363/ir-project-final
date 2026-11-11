"""Evaluate on EXPERT labels you create.  Usage (from backend/): python eval_expert.py ../data/real/expert_labels.csv
CSV columns: query_id, query_text, case_id, issue   (one row per case/issue pair a lawyer judged relevant; case_id = id in real_cases.json)
Cases not listed for a query count as irrelevant. Aim for 30+ queries with 3+ issues each."""
import sys, collections, numpy as np, pandas as pd
from run_real import build, search, tok
from irdiv.retrieval import InvertedIndex, TfidfSpace
from irdiv.metrics import Evaluator

lab = pd.read_csv(sys.argv[1]); docs = build(); idx = InvertedIndex(docs)
for d in docs: d["text"] = d["text_k"]
tf = TfidfSpace(docs); rows = []
for qid, g in lab.groupby("query_id"):
    issues = sorted(g.issue.unique()); m = collections.defaultdict(set)
    for _, r in g.iterrows(): m[int(r.case_id)].add(issues.index(r.issue))
    qd = [dict(d, issues=tuple(sorted(m.get(d["id"], ())))) for d in docs]
    ev = Evaluator(qd, list(range(len(issues))), 10); _, _, _, bm, mm, sb = search(docs, idx, tf, tok(g.query_text.iloc[0]), 10, 30, 0.7)
    for n, ids in (("BM25", bm), ("MMR", mm), ("Submodular", sb)): rows.append(dict(query_id=qid, system=n, **ev.evaluate(ids)))
df = pd.DataFrame(rows); print(df.groupby("system")[["P", "S_recall", "alpha_NDCG", "NDCG"]].mean().round(3))
