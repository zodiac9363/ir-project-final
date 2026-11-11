"""Small LLM layer. Uses a local model through Ollama (e.g. qwen2.5:1.5b) when it is running,
otherwise a deterministic offline template so the pipeline still works and is testable.
The LLM never ranks or selects: it only (1) explains a chosen set and (2) rewrites synthetic cases into
natural sentences for a realism check. Prompts contain only facts computed by the pipeline."""
import json, os, urllib.request


class OfflineLLM:
    name, offline = "offline-template (no LLM)", True

    def generate(self, prompt):
        return None


class OllamaLLM:
    offline = False

    def __init__(self, model="qwen2.5:1.5b", host="http://localhost:11434"):
        self.model, self.host, self.name = model, host, f"ollama:{model}"

    def available(self):
        try:
            req = urllib.request.urlopen(self.host + "/api/tags", timeout=1.5)
            data = json.loads(req.read().decode())
            installed = [m.get("name") for m in data.get("models", [])]
            if not installed:
                return False
            # If specified model is available, use it
            if any(self.model in name for name in installed):
                return True
            # Otherwise pick the best installed model (e.g. qwen2.5:3b)
            self.model = installed[0]
            self.name = f"ollama:{self.model}"
            return True
        except Exception:
            return False

    def generate(self, prompt, timeout=120):
        try:
            body = json.dumps(dict(model=self.model, prompt=prompt, stream=False, options=dict(temperature=0))).encode()
            req = urllib.request.Request(self.host + "/api/generate", body, {"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["response"].strip()
        except Exception:
            return None


def get_llm(model=None):
    o = OllamaLLM(model or os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b"))
    return o if o.available() else OfflineLLM()


def _facts(q_issues, picks):
    covered, out = [], []
    for i, p in enumerate(picks):
        new = [x for x in p["issues"] if x in q_issues and x not in covered]
        covered += new
        out.append(dict(pos=i + 1, id=p["id"], issues=p["issues"], new=new, dup=p.get("dup", False), root=p.get("root"), rank=p.get("rank")))
    return out, covered


def explain_selection(llm, q_name, q_issues, picks):
    facts, covered = _facts(q_issues, picks)
    missing = [x for x in q_issues if x not in covered]
    lines = [f"- pick {f['pos']}: case #{f['id']} ({', '.join(f['issues'])}), BM25 rank {f['rank']}, "
             f"new query issues: {', '.join(f['new']) or 'none'}" + (f", near-copy of #{f['root']}" if f['dup'] else "") for f in facts]
    if not llm.offline:
        prompt = (f"You help a lawyer. Query issues: {', '.join(q_issues)}.\nSelected cases:\n" + "\n".join(lines) +
                  f"\nIssues not covered: {', '.join(missing) or 'none'}.\nIn 3 sentences, explain why this set suits the query. "
                  "Use only the facts above. Do not invent case details.")
        try:
            return llm.generate(prompt)
        except Exception:
            pass
    t = [f"The {len(picks)} cases cover {len(covered)} of {len(q_issues)} query issues ({', '.join(covered) or 'none'})."]
    for f in facts[:5]:
        if f["pos"] == 1:
            t.append(f"Case #{f['id']} ({', '.join(f['issues'])}) is the closest single match.")
        elif f["new"]:
            t.append(f"Case #{f['id']} adds {', '.join(f['new'])}, which no earlier pick covered.")
        elif f["dup"]:
            t.append(f"Case #{f['id']} is a near-copy of #{f['root']}.")
    if missing:
        t.append(f"Not covered: {', '.join(missing)}.")
    return " ".join(t)


def rewrite_case(llm, words, issues):
    """Turn a bag-of-words synthetic case into a short natural summary (needs a real LLM; else returns None)."""
    if llm.offline:
        return None
    return llm.generate("Write a 3-sentence court-judgment summary about: " + ", ".join(issues) +
                        ". Naturally use some of these words: " + ", ".join(sorted(set(words))[:15]) + ". Do not add other legal topics.")
