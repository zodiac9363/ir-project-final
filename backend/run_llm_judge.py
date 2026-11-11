"""LLM-as-judge on real cases.  Usage (from backend/), Ollama running:  python run_llm_judge.py
For 5 plain-language problems, BM25 / MMR / Submodular each return 10 cases. A local LLM grades every (problem, case) pair
0-2 and says which of the problem's issues the case deals with. The LLM only judges; it never ranks.
Writes ../results/llm_judge.csv and llm_judge_agreement.csv. Bias warning: LLM judges are imperfect; report them next to, not instead of, human labels."""
import sys, json, numpy as np, pandas as pd
from run_real import build, search, tok
from irdiv.retrieval import InvertedIndex, TfidfSpace
from irdiv.llm import get_llm
from irdiv.llm_tools import judge_all, kappa

PROBLEMS = [
    ("Workers were retrenched by their employer, who also disputes its sales tax assessment, and the workers say their right to equality was violated.", ["Labour and industrial disputes", "Tax", "Constitutional rights"]),
    ("A landlord wants to evict a tenant for unpaid rent, and the dispute also involves a partnership agreement over the shop and rights over the land.", ["Tenancy and eviction", "Land and property", "Company, partnership and contract"]),
    ("A government employee was dismissed after a disciplinary inquiry, is also being prosecuted for an offence, and claims his fundamental rights were violated.", ["Service and employment", "Criminal law", "Constitutional rights"]),
    ("A taxpayer disputes an income tax assessment.", ["Tax"]),
    ("A person convicted of murder appeals against the conviction and sentence.", ["Criminal law"]),
]


def main(llm=None, out="../results", cache="../data/real/judge_cache.json", raw=None):
    llm = llm or get_llm()
    if llm.offline:
        sys.exit("No LLM found. Install Ollama and run:  ollama pull qwen2.5:1.5b   (see docs/OLLAMA_SETUP.md)")
    docs = build(raw); excerpt = {d["id"]: d["text"] for d in docs}; idx = InvertedIndex(docs)
    for d in docs: d["text"] = d["text_k"]
    tf = TfidfSpace(docs); runs, items = {}, []
    for qi, (prob, iss) in enumerate(PROBLEMS):
        _, _, _, bm, mm, sb = search(docs, idx, tf, tok(prob), 10, 30, 0.7)
        for n, ids in (("BM25", bm), ("MMR", mm), ("Submodular", sb)):
            runs[(qi, n)] = [int(i) for i in ids]
            items += [(f"{qi}|{i}", prob, iss, excerpt[int(i)]) for i in ids]
    J = judge_all(llm, list({it[0]: it for it in items}.values()), cache)
    rows, pairs = [], []
    for (qi, n), ids in runs.items():
        iss = PROBLEMS[qi][1]; g = [J[f"{qi}|{i}"] for i in ids]; ok = [x for x in g if x]
        cov = set(t for x in ok if x["grade"] >= 1 for t in x["issues"])
        rows.append(dict(query=qi + 1, system=n, judged=len(ok), P_judge=np.mean([x["grade"] >= 1 for x in ok]) if ok else np.nan,
                         mean_grade=np.mean([x["grade"] for x in ok]) if ok else np.nan, S_recall_judge=len(cov) / len(iss)))
        for i, x in zip(ids, g):
            if x: pairs.append((x["grade"] >= 1, bool(set(docs[i]["topics"]) & set(iss))))
    df = pd.DataFrame(rows); m = df.groupby("system")[["P_judge", "mean_grade", "S_recall_judge"]].mean().round(3).reset_index()
    m.to_csv(f"{out}/llm_judge.csv", index=False); print(m.to_string(index=False))
    if len(pairs) > 5:
        a, b = zip(*pairs); k = pd.DataFrame([dict(pairs=len(pairs), cohen_kappa_judge_vs_rule_labels=round(kappa(a, b), 3))])
        k.to_csv(f"{out}/llm_judge_agreement.csv", index=False); print(k.to_string(index=False))


if __name__ == "__main__":
    main()
