"""Synthetic legal-case corpus with known ground truth (issues, near-copies, qrels)."""
import numpy as np

_W = lambda s: s.split()
ISSUES = [
    ("Late payment", _W("payment late overdue invoice arrears default installment interest remit delay")),
    ("Wrongful termination", _W("terminate dismissal employee notice severance employer retaliation discharge tenure redundancy")),
    ("Penalty clause", _W("penalty liquidated damages forfeiture excessive punitive stipulated unconscionable surcharge deterrent")),
    ("Breach of warranty", _W("warranty defect goods merchantable fitness quality defective replacement guarantee recall")),
    ("Misrepresentation", _W("misrepresent fraudulent concealment inducement rescission disclosure deceit falsehood reliance statement")),
    ("Jurisdiction", _W("jurisdiction venue forum territorial arbitration seat competent tribunal domicile service")),
]
GENERAL = _W("contract court party agreement plaintiff defendant clause breach held appeal judgment liability terms obligation relief remedy section")
ISSUE_WEIGHTS = np.array([.35, .15, .12, .15, .12, .11])

# reps = how many words of each issue's vocabulary go into the query
QUERIES = [
    dict(name="late payment + termination + penalty", issues=[0, 1, 2], reps=[8, 1, 1], control=False),
    dict(name="late payment + warranty + misrepresentation", issues=[0, 3, 4], reps=[8, 1, 1], control=False),
    dict(name="late payment + termination + jurisdiction", issues=[0, 1, 5], reps=[8, 1, 1], control=False),
    dict(name="CONTROL single issue: warranty", issues=[3], reps=[4], control=True),
    dict(name="CONTROL single issue: late payment", issues=[0], reps=[4], control=True),
]


def query_terms(q):
    t = []
    for i, r in zip(q["issues"], q["reps"]):
        t += ISSUES[i][1][:r]
    return t


def make_corpus(n=400, dup_rate=0.3, seed=0, words_per_doc=40):
    """Each doc: 1-2 issues, 70% issue words / 30% general words.
    With prob dup_rate a doc is a near-copy (10% words swapped) of an earlier doc."""
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        if i > 20 and rng.random() < dup_rate:
            src = docs[rng.integers(len(docs))]
            words = [w if rng.random() > .1 else str(rng.choice(GENERAL)) for w in src["words"]]
            docs.append(dict(id=i, words=words, issues=src["issues"], root=src["root"], dup=True))
        else:
            c = int(rng.choice(6, p=ISSUE_WEIGHTS))
            iss = [c]
            if rng.random() < .35:
                iss.append(int(rng.choice([x for x in range(6) if x != c])))
            words = [str(rng.choice(ISSUES[int(rng.choice(iss))][1])) if rng.random() < .7
                     else str(rng.choice(GENERAL)) for _ in range(words_per_doc)]
            docs.append(dict(id=i, words=words, issues=tuple(iss), root=i, dup=False))
    for d in docs:
        d["text"] = " ".join(d["words"])
    return docs
