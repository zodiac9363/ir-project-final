"""LLM issue labelling for the real cases, with a hand-check.  Usage (from backend/), Ollama must be running:
  python run_llm_labels.py label     # LLM labels every case      -> ../data/real/llm_labels.json
  python run_llm_labels.py sample    # 40 random cases to hand-label -> ../data/real/handcheck.csv  (fill 'correct_issues', separate with ;)
  python run_llm_labels.py score     # agreement of LLM labels and rule labels with YOUR labels -> ../results/label_agreement.csv
Then:  python run_real.py --labels llm   evaluates the real cases using the LLM labels."""
import sys, json, os, random
import pandas as pd
from irdiv.real import load_real, snippet, TOPICS
from irdiv.llm import get_llm
from irdiv.llm_tools import label_all, prf

RAW, LAB, HC = "../data/real/cases_raw.json", "../data/real/llm_labels.json", "../data/real/handcheck.csv"
TN = list(TOPICS)


def main(cmd, llm=None, raw=RAW, lab=LAB, hc=HC, out="../results"):
    cases = load_real(raw)
    if cmd == "label":
        llm = llm or get_llm()
        if llm.offline:
            sys.exit("No LLM found. Install Ollama and run:  ollama pull qwen2.5:1.5b   (see docs/OLLAMA_SETUP.md)")
        c = label_all(llm, cases, TN, lab); print(sum(1 for v in c.values() if v["llm"]), "of", len(c), "cases labelled; saved to", lab)
    elif cmd == "sample":
        c = json.load(open(lab)) if os.path.exists(lab) else {}
        ids = sorted(random.Random(0).sample(range(len(cases)), min(40, len(cases))))
        pd.DataFrame([dict(case_id=i, excerpt=snippet(cases[i]["text"], 600), llm_issues=";".join((c.get(str(i), {}).get("llm") or [])),
                           rule_issues=";".join(cases[i]["topics"]), correct_issues="") for i in ids]).to_csv(hc, index=False)
        print("Wrote", hc, "- fill the correct_issues column using these names:", TN)
    elif cmd == "score":
        d = pd.read_csv(hc).fillna(""); d = d[d.correct_issues != ""]
        if d.empty:
            sys.exit("Fill the correct_issues column in " + hc + " first.")
        sp = lambda col: [[x for x in s.split(";") if x] for s in d[col]]
        r = pd.DataFrame([dict(source=n, cases_checked=len(d), **prf(sp(col), sp("correct_issues"))) for n, col in (("LLM labels", "llm_issues"), ("Keyword-rule labels", "rule_issues"))]).round(3)
        r.to_csv(f"{out}/label_agreement.csv", index=False); print(r.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "label")
