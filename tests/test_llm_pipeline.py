"""Pipeline test with a STAND-IN model (keyword rules posing as an LLM). It checks parsing, caching, hand-check scoring and the judge
flow. It says nothing about how good a real LLM is. Run from backend/:  python ../tests/test_llm_pipeline.py"""
import json, os, re, sys, tempfile
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
import pandas as pd
from irdiv.real import _PAT
import run_llm_labels as L, run_llm_judge as J


class FakeLLM:
    offline, name, calls = False, "fake-test-double", 0

    def generate(self, prompt):
        FakeLLM.calls += 1
        if prompt.startswith("You label"):
            ex = prompt.split('"""')[1]; hits = sorted(((len(p.findall(ex)), t) for t, p in _PAT.items()), reverse=True)[:2]
            return "Sure! " + json.dumps({"issues": [t for h, t in hits if h > 3]})        # chatty wrapper tests JSON extraction
        iss = json.loads(re.search(r"Issues in their problem: (\[.*\])", prompt).group(1)); ex = prompt.split('"""')[1]
        got = [t for t in iss if len(_PAT[t].findall(ex)) > 3]
        return json.dumps({"grade": 2 if len(got) > 1 else (1 if got else 0), "issues_addressed": got})


tmp = tempfile.mkdtemp(); lab, hc, cache = f"{tmp}/l.json", f"{tmp}/h.csv", f"{tmp}/c.json"
L.main("label", FakeLLM(), lab=lab); n1 = FakeLLM.calls; L.main("label", FakeLLM(), lab=lab)
assert FakeLLM.calls == n1, "second run must use the cache"
L.main("sample", lab=lab, hc=hc); d = pd.read_csv(hc).fillna(""); assert len(d) == 40
d["correct_issues"] = d.rule_issues; d.to_csv(hc, index=False)                              # pretend a human agreed with the rules
L.main("score", hc=hc, out=tmp); print(pd.read_csv(f"{tmp}/label_agreement.csv").to_string(index=False))
J.main(FakeLLM(), out=tmp, cache=cache); assert os.path.exists(f"{tmp}/llm_judge.csv")
print("TEST OK (stand-in model only; no real results produced)")
