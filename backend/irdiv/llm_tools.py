"""LLM helpers for (1) labelling real cases with legal issues and (2) judging search results.
The LLM is used as an annotator, never as a ranker. Output is validated against fixed lists, and every call is cached on disk."""
import json, os, re
import numpy as np


def _json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def label_case(llm, text, topics, max_n=3):
    p = ("You label Indian Supreme Court judgments. Read the excerpt and choose up to 3 legal issues it is mainly about, "
         "ONLY from this list: " + json.dumps(topics) + '\nExcerpt:\n"""' + text[:3000] + '"""\nReturn ONLY JSON like {"issues": ["..."]}')
    j = _json(llm.generate(p))
    out = [t for t in (j or {}).get("issues", []) if t in topics][:max_n] if isinstance(j, dict) else []
    return out or None


def judge_pair(llm, problem, issues, excerpt):
    p = ("A person has this legal problem:\n" + problem + "\nIssues in their problem: " + json.dumps(issues) +
         '\nJudge ONLY from this judgment excerpt:\n"""' + excerpt[:2500] + '"""\n'
         'Return ONLY JSON: {"grade": 0, "issues_addressed": []} where grade is 0 (not relevant), 1 (partly relevant) or 2 (directly relevant), '
         "and issues_addressed lists which of the problem's issues the excerpt deals with.")
    j = _json(llm.generate(p))
    if not isinstance(j, dict) or j.get("grade") not in (0, 1, 2):
        return None
    return dict(grade=int(j["grade"]), issues=[x for x in j.get("issues_addressed", []) if x in issues])


def _cached(path, key, fn, cache):
    if key not in cache:
        cache[key] = fn()
        json.dump(cache, open(path, "w"))
    return cache[key]


def label_all(llm, cases, topics, path):
    cache = json.load(open(path)) if os.path.exists(path) else {}
    for c in cases:
        _cached(path, str(c["id"]), lambda c=c: dict(llm=label_case(llm, c["text"], topics), rules=c["topics"]), cache)
    return cache


def judge_all(llm, items, path):
    """items: list of (key, problem, issues, excerpt)."""
    cache = json.load(open(path)) if os.path.exists(path) else {}
    for key, problem, issues, excerpt in items:
        _cached(path, key, lambda: judge_pair(llm, problem, issues, excerpt), cache)
    return cache


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def prf(pred, gold):
    """Micro precision / recall / F1 and exact-match rate over lists of label sets."""
    tp = sum(len(set(p) & set(g)) for p, g in zip(pred, gold)); np_ = sum(len(set(p)) for p in pred); ng = sum(len(set(g)) for g in gold)
    pr, rc = tp / max(np_, 1), tp / max(ng, 1)
    return dict(precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-9), exact_match=float(np.mean([set(p) == set(g) for p, g in zip(pred, gold)])))
