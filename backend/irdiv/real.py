"""Real-case loader: Indian Supreme Court judgments -> same format as the synthetic corpus.
Issue labels are AUTOMATIC (keyword topic lexicon + statutes cited), so they are weak labels, not expert judgments."""
import json, re
from collections import Counter

TOPICS = {  # topic -> regex of indicative phrases
    "Tenancy and eviction": r"\b(tenant|landlord|eviction|rent control|bona fide requirement|lessee|lessor|tenancy|arrears of rent)\b",
    "Land and property": r"\b(land acquisition|compensation|acquisition|title|possession|sale deed|mortgage|partition|ceiling|easement|trespass)\b",
    "Criminal law": r"\b(accused|murder|conviction|sentence|prosecution|offence|bail|police|penal code|cognizable|criminal appeal)\b",
    "Tax": r"\b(income[- ]?tax|sales tax|assessee|assessment|excise|customs|turnover|tax(?:able|ation)?)\b",
    "Labour and industrial disputes": r"\b(workman|workmen|industrial dispute|tribunal|retrenchment|lock-?out|strike|bonus|wages|trade union)\b",
    "Constitutional rights": r"\b(article 14|article 19|article 21|article 226|article 32|fundamental right|writ petition|constitution of india|discriminat\w+)\b",
    "Service and employment": r"\b(promotion|seniority|dismissal from service|civil servant|disciplinary|pension|public servant|termination of service)\b",
    "Company, partnership and contract": r"\b(partnership|company|shareholder|contract|agreement|breach|specific performance|damages|guarantee)\b",
    "Evidence and procedure": r"\b(evidence act|limitation act|res judicata|code of civil procedure|review petition|burden of proof|admissib\w+|estoppel)\b",
    "Corruption and public office": r"\b(corruption|bribe|public servant|prevention of corruption|illegal gratification)\b",
}
_PAT = {t: re.compile(p, re.I) for t, p in TOPICS.items()}
_ACT = re.compile(r"\b((?:[A-Z][A-Za-z\.&']+\s+){0,5}[A-Z][A-Za-z\.&']+\s+Act)\b")


_VS = re.compile(r"([A-Z][A-Za-z\.&' ]{2,45}?)\s+(?:v\.|vs\.?|versus)\s+([A-Z][A-Za-z\.&' ,]{2,50}?)(?=[\s,\.\)]| and| \()")


def _title(text, fallback):
    m = _VS.search(text[:3000])
    return clean(f"{m.group(1)} v. {m.group(2)}")[:90] if m else fallback[:90]


def snippet(text, n=420):
    m = re.search(r"(?:delivered by|Judgment of the Court was)[^.]{0,80}\.\s*", text[:6000])
    return text[m.end():m.end() + n] if m else text[300:300 + n]


def clean(t):
    return re.sub(r"\s+", " ", t).strip()


def load_real(path=None, min_chars=2000, min_hits=6):
    """Keeps full judgments, tags topics with >= min_hits keyword hits per 10k chars scaled.
    Automatically prioritizes the downloaded ILDC dataset (Exploration-Lab/CJPE) when available."""
    import os
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(base_dir, "../../data/real"))
    ildc_path = os.path.join(data_dir, "ildc_cases.json")
    raw_path = os.path.join(data_dir, "cases_raw.json")

    if path is None or not os.path.exists(path):
        path = ildc_path if os.path.exists(ildc_path) else raw_path

    out = []
    for c in json.load(open(path, encoding="utf-8")):
        if len(c.get("text", "")) < min_chars:
            continue
        text = clean(c["text"])
        n = max(len(text) / 10000, 1)
        hits = {t: len(p.findall(text)) for t, p in _PAT.items()}
        topics = [t for t, h in sorted(hits.items(), key=lambda x: -x[1]) if h >= min_hits * min(n, 3) / 3 and h >= 4][:3]
        acts = [a for a, _ in Counter(clean(m) for m in _ACT.findall(text)).most_common(3) if 2 < len(a.split()) <= 8]
        fallback_name = c.get("case_name") or f"Case #{c.get('id', len(out))}"
        title = _title(text, clean(fallback_name))
        out.append(dict(id=len(out), orig_id=c["id"], title=title, text=text, topics=topics or ["General Legal"], acts=acts, root=len(out)))
    return out
