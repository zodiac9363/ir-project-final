"""Real Indian Supreme Court judgments.  Usage (from backend/):
  python run_real.py                               # weak-label evaluation + export for the web page
  python run_real.py --query "my landlord evicted me without notice and..." --k 5   # search from the command line
Labels are automatic topic tags (weak labels), so scores here are indicative only."""
import argparse, json, re, numpy as np, pandas as pd
from irdiv.real import load_real, snippet, TOPICS
from irdiv.retrieval import InvertedIndex, TfidfSpace
from irdiv.diversify import greedy, mmr
from irdiv.metrics import Evaluator
from irdiv.llm import get_llm

STOP = set("the of and to in a an is was were for on by with that this it as at be from or not which have has had are been their its his her they he she them under upon any such but if no shall may can also than these those said".split())
tok = lambda t: [w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP and len(w) > 2]
TN = list(TOPICS)
QW = {"Tenancy and eviction": "tenant landlord eviction rent tenancy", "Land and property": "land possession title acquisition compensation",
      "Criminal law": "accused conviction offence sentence prosecution", "Tax": "assessee tax assessment sales income",
      "Labour and industrial disputes": "workmen industrial tribunal retrenchment wages", "Constitutional rights": "fundamental constitution article writ discrimination",
      "Service and employment": "promotion seniority service dismissal disciplinary", "Company, partnership and contract": "partnership company contract agreement breach"}
QUERIES = [["Tax", "Labour and industrial disputes", "Constitutional rights"], ["Land and property", "Tenancy and eviction", "Company, partnership and contract"],
           ["Criminal law", "Service and employment", "Constitutional rights"], ["Tax"], ["Criminal law"]]


def build(path=None, labels=None):
    docs = load_real(path)
    if labels:  # override topics with LLM (or hand) labels where available
        for d in docs:
            t = (labels.get(str(d["id"])) or {}).get("llm")
            if t: d["topics"] = t
    for d in docs:
        d["words"] = tok(d["text"]); d["issues"] = tuple(TN.index(t) for t in d["topics"] if t in TN); d["dup"] = False
        d["text_k"] = " ".join(d["words"])
    return docs


def search(docs, idx, tf, terms, k=10, pool=30, lam=0.7, mode="saturated"):
    s = idx.bm25(terms); order = np.lexsort((np.arange(len(s)), -s))[:pool]; r = s[order] / (s[order].max() or 1)
    S = tf.sim(order)
    sub_picks = greedy(r, S, k, lam, mode=mode)[0]
    return order, r, S, order[:k], order[mmr(r, S, k, lam)], order[sub_picks]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--query"); ap.add_argument("--k", type=int, default=10); ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--out", default="../results"); ap.add_argument("--labels", default="rules", choices=["rules", "llm"]); a = ap.parse_args()
    docs = build(labels=json.load(open("../data/real/llm_labels.json")) if a.labels == "llm" else None); idx = InvertedIndex(docs)
    for d in docs: d["text"] = d["text_k"]
    tf = TfidfSpace(docs)
    if a.query:
        llm = get_llm(); terms = tok(a.query)
        if not llm.offline:
            try: terms += tok(llm.generate("List 8 search keywords (single words, comma separated) for this legal problem: " + a.query))
            except Exception: pass
        _, _, _, _, _, sub = search(docs, idx, tf, terms, a.k, 30, a.lam)
        for i in sub: print(f"#{i}  {docs[i]['title']}  {docs[i]['topics']}")
        return
    rows = []
    for qi, topics in enumerate(QUERIES):
        terms = (QW[topics[0]].split() + sum([QW[t].split()[:2] for t in topics[1:]], [])); q = [TN.index(t) for t in topics]
        ev = Evaluator(docs, q, 10); _, _, _, bm, mm, sb = search(docs, idx, tf, terms, 10, 30, a.lam)
        for n, ids in (("BM25", bm), ("MMR", mm), ("Submodular", sb)):
            rows.append(dict(query=" + ".join(topics), system=n, **ev.evaluate(ids)))
    df = pd.DataFrame(rows); sfx = "" if a.labels == "rules" else "_llmlabels"; df.round(3).to_csv(f"{a.out}/real_eval_queries{sfx}.csv", index=False)
    m = df.groupby("system")[["P", "S_recall", "alpha_NDCG", "NDCG", "Dups"]].mean().round(3).reset_index(); m.to_csv(f"{a.out}/real_eval{sfx}.csv", index=False); print(m.to_string(index=False))
    if a.labels != "rules": return
    ref = lambda t: re.sub(r"\s+", " ", t)[:34]
    real_docs = load_real()
    cases = [dict(id=d["id"], title=(ref(d["title"]) + " | " + snippet(d["text"])[:70]).strip(), topics=d["topics"], acts=d["acts"], snip=snippet(d["text"]), text=d["text"][:9000]) for d in real_docs]
    json.dump(dict(topics=TN, eval=m.to_dict("records"), cases=cases), open("../data/real/real_cases.json", "w"), separators=(",", ":")); print(len(cases), "cases exported")


if __name__ == "__main__":
    main()
