# Running a local LLM with Ollama

The project uses a small local model for three jobs: explaining result sets, labelling the real cases with legal issues, and judging search results. It never ranks. Everything runs offline once the model is downloaded.

## 1. Install
1. Download Ollama from https://ollama.com and install it (Windows, Mac and Linux are supported).
2. Open a terminal and download a model:
   `ollama pull qwen2.5:1.5b`
   The 1.5B model is small and fast. For better labels on a laptop with enough memory, try `qwen2.5:3b` or `qwen2.5:7b` and set the environment variable `OLLAMA_MODEL` to the name you pulled (for example `set OLLAMA_MODEL=qwen2.5:3b` on Windows, `export OLLAMA_MODEL=qwen2.5:3b` on Mac/Linux). Larger models need more RAM and are slower.
3. Check it is running: open http://localhost:11434/api/tags in a browser. You should see your model listed. If not, start Ollama from your applications menu or run `ollama serve`.

## 2. Run the LLM steps (from the `backend/` folder)
    python run_llm_labels.py label     # LLM labels all 100 real cases (cached, safe to stop and resume)
    python run_llm_labels.py sample    # writes data/real/handcheck.csv with 40 random cases
    (open handcheck.csv, fill the correct_issues column with names from the list it prints, separated by ;)
    python run_llm_labels.py score     # how often LLM labels and keyword labels agree with YOU
    python run_real.py --labels llm    # re-run the real-case evaluation using the LLM labels
    python run_llm_judge.py            # LLM grades the results of BM25, MMR and submodular
    cd .. && python build.py           # refresh the web page with the new tables

## 3. What to report in the viva
- The label agreement table is your measured label quality. Say it plainly, for example: "the LLM labels agreed with my hand labels with F1 = x on 40 cases, the keyword rules with F1 = y."
- The LLM judge is a second opinion, not ground truth. Report its agreement with the keyword labels (Cohen's kappa) and note that small models are noisy and can have systematic biases.
- The LLM never ranks or selects cases, so differences between BM25, MMR and submodular selection come from the selection method alone.

## Troubleshooting
- "No LLM found": Ollama is not running or the model name does not match. Run `ollama list` and check `OLLAMA_MODEL`.
- Slow or timing out: use the smaller model, or close other heavy programs. Progress is saved after each call, so re-running continues where it stopped.
- Many cases come back unlabelled: the model is returning text that is not valid JSON. Try a larger model.
- `tests/test_llm_pipeline.py` checks the plumbing with a stand-in model. It needs no Ollama and says nothing about real model quality.
