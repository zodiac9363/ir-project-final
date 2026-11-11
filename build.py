"""One command rebuilds everything:  python build.py [--seeds 10]
 1. runs experiments E1-E10 + vector-DB/LLM stage E11 (backend) -> results/*.csv, figures
 2. exports a pipeline trace + tables   -> data/frontend_data.json
 3. embeds that JSON into the front end -> frontend/index.html (the 'Python pipeline' tab)"""
import argparse, pathlib, re, subprocess, sys
root = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); a = ap.parse_args()
run = lambda *c: subprocess.run([sys.executable, *c], cwd=root / "backend", check=True)
run("run_experiments.py", "--seeds", str(a.seeds), "--out", "../results")
run("run_vectordb.py", "--seeds", str(a.seeds), "--out", "../results")
run("run_real.py")
run("eval_ildc_expert.py")
run("run_extras.py", "--seeds", str(a.seeds), "--out", "../results")
run("export_frontend.py")
data = (root / "data" / "frontend_data.json").read_text(encoding="utf-8")
page = root / "frontend" / "index.html"
html = page.read_text(encoding="utf-8")
html, n = re.subn(r'(<script id="pydata" type="application/json">).*?(</script>)', lambda m: m.group(1) + data + m.group(2), html, count=1, flags=re.S)
assert n == 1, "pydata block not found in frontend/index.html"
real = (root / "data" / "real" / "real_cases.json").read_text(encoding="utf-8")
html, n2 = re.subn(r'(<script id="realdata" type="application/json">).*?(</script>)', lambda m: m.group(1) + real + m.group(2), html, count=1, flags=re.S)
assert n2 == 1, "realdata block not found"
page.write_text(html, encoding="utf-8")
print("Front end updated. Open it with:  python serve.py")
