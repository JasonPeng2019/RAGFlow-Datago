alphago_project venv – How to use this environment

What this venv is for
- Lightweight Python tooling around this workspace (scripts/clients, quick data collection, API calls).
- You do NOT need this venv to run KataGo binaries (they are C++). You also do NOT need it to run RAGFlow (Docker-based).

Activate
  source /scratch2/f004h1v/alphago_project/venv/bin/activate
  python -V

Common tasks

1) Talk to KataGo’s JSON analysis engine
- Start the engine in another shell (GPU/OpenCL):
  cd /scratch2/f004h1v/alphago_project/katago_repo/run
  ../KataGo/cpp/build-opencl/katago analysis -model default_model.bin.gz -config analysis.cfg

- From this venv, you can write small Python clients to send/receive JSON. See upstream example:
  /scratch2/f004h1v/alphago_project/katago_repo/KataGo/python/query_analysis_engine_example.py

- Useful installs (if you build clients):
  pip install numpy requests

2) Run KataGo GTP quick tests (no Python needed, but handy to orchestrate)
- Engine (short run):
  cd /scratch2/f004h1v/alphago_project/katago_repo/run
  ../KataGo/cpp/build-opencl/katago gtp -model default_model.bin.gz -config default_gtp.cfg -override-config maxVisits=100

- Logs and configs:
  - default_gtp.cfg: main GTP config (threads/visits/rules)
  - gtp_logs/: session logs
  - default_model.bin.gz: symlink to the current model file

3) Simple data collection ideas
- Use the analysis engine to dump policy/value/ownership for SGF positions.
- Log policy entropy per position to decide when to trigger RAG (root entropy).
- Store outputs to CSV/JSON via small Python scripts in this venv.

RAGFlow (separate stack)
- RAGFlow is Docker-first and does NOT run inside this venv.
- To start it, follow:
  /scratch2/f004h1v/alphago_project/ragflow_repo/README.md

Where everything is
- KataGo binaries (GPU/OpenCL):
  /scratch2/f004h1v/alphago_project/katago_repo/KataGo/cpp/build-opencl/katago
- Runtime folder (models/configs/logs):
  /scratch2/f004h1v/alphago_project/katago_repo/run
- RAG project workspace (our code to add):
  /scratch2/f004h1v/alphago_project/datago
- RAGFlow upstream clone and docs:
  /scratch2/f004h1v/alphago_project/ragflow_repo

Tips
- GPU selection/tuning for KataGo is handled by config and first-time autotuning; see run/default_gtp.cfg.
- If you need more Python libs for experiments, install them here to keep the system clean.

References
- KataGo upstream: https://github.com/lightvector/KataGo
- RAGFlow upstream: https://github.com/infiniflow/ragflow


