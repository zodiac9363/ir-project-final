"""Dual-purpose server: Serves the web frontend and hosts a live Python Search API.
Run:  python serve.py
Open: http://localhost:8000"""
import http.server, json, os, re, sys, urllib.parse, webbrowser

# Add backend directory to sys.path so backend modules load cleanly
root = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from irdiv.real import load_real, snippet, TOPICS
from irdiv.retrieval import InvertedIndex, TfidfSpace
from irdiv.diversify import greedy, mmr
from irdiv.llm import get_llm
from run_real import build, search, tok

print("Initializing Legal Retrieval Engine on real judgments...")
real_docs = build()
idx = InvertedIndex(real_docs)
for d in real_docs:
    d["text"] = d["text_k"]
tf = TfidfSpace(real_docs)
llm = get_llm()
print(f"Engine Ready: {len(real_docs)} Supreme Court judgments indexed.")
print(f"LLM Status: {llm.name} (Active: {not llm.offline})")

FRONTEND_DIR = os.path.join(root, "frontend")

class LegalSearchHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            info = {
                "status": "online",
                "dataset": "ILDC (Exploration-Lab/CJPE) Supreme Court Corpus",
                "indexed_cases": len(real_docs),
                "topics": list(TOPICS.keys()),
                "llm": llm.name,
                "llm_connected": not llm.offline
            }
            self.wfile.write(json.dumps(info).encode("utf-8"))
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/search":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
            except Exception:
                data = {}

            query = data.get("query", "").strip()
            k = int(data.get("k", 5))
            lam = float(data.get("lambda", 0.7))
            terms = tok(query)

            if not terms:
                terms = ["law", "court", "appeal"]

            order, r, S, bm, mm, sb = search(real_docs, idx, tf, terms, k=k, pool=40, lam=lam, mode="saturated")

            ref = lambda t: re.sub(r"\s+", " ", t)[:40]
            format_case = lambda i, rk: {
                "id": real_docs[i]["id"],
                "title": real_docs[i]["title"],
                "topics": real_docs[i]["topics"],
                "acts": real_docs[i]["acts"],
                "snip": snippet(real_docs[i]["text"], n=350),
                "rank": rk,
                "score": round(float(r[rk - 1]), 3) if rk <= len(r) else 0.0
            }

            rec = [format_case(idx_id, pos + 1) for pos, idx_id in enumerate(sb)]
            bm_cases = [format_case(idx_id, pos + 1) for pos, idx_id in enumerate(bm)]
            mm_cases = [format_case(idx_id, pos + 1) for pos, idx_id in enumerate(mm)]

            llm_explanation = None
            if not llm.offline and query:
                try:
                    summary_prompt = (
                        f"You are an expert legal assistant. A user researched: '{query}'.\n"
                        f"The submodular retriever selected these varied Supreme Court cases covering distinct angles:\n"
                        + "\n".join([f"- {c['title']} (Topics: {', '.join(c['topics'])})" for c in rec[:3]]) +
                        "\nIn exactly 2 concise sentences, summarize why having these different angles is valuable for this legal research."
                    )
                    llm_explanation = llm.generate(summary_prompt, timeout=15)
                except Exception:
                    llm_explanation = None

            resp_data = {
                "status": "success",
                "query": query,
                "terms": terms,
                "dataset": "ILDC / Supreme Court of India",
                "llm": llm.name,
                "llm_connected": not llm.offline,
                "llm_explanation": llm_explanation,
                "recommended": rec,
                "bm25": bm_cases,
                "mmr": mm_cases
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def main():
    port = 8000
    server = http.server.ThreadingHTTPServer(("", port), LegalSearchHandler)
    print(f"\n=======================================================")
    print(f" Live Legal Search Server running at http://localhost:{port}")
    print(f" Serving Frontend UI + Live Python /api/search API")
    print(f" Dataset: ILDC (Exploration-Lab/CJPE) & Indian Supreme Court")
    print(f" Press Ctrl+C to stop.")
    print(f"=======================================================\n")
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")

if __name__ == "__main__":
    main()
