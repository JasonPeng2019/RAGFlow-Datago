alphago_project – Workspace Guide

This document explains every directory and key file in this workspace, how to run KataGo on GPU, where configs and models live, and how we will organize the new RAG-augmented research under datago.

Contents
- Project layout
- Quick start: run KataGo (GPU)
- Files and folders in depth
- How configs and logs work
- Python environment
- Next steps for the RAG project (datago)

Project layout

  /scratch2/f004h1v/alphago_project
  ├─ katago_repo/                # The KataGo repository and builds
  │  ├─ KataGo/                 # Upstream repo source tree (C++/Python/docs)
  │  │  └─ cpp/
  │  │     ├─ build-opencl/     # GPU (OpenCL) build; contains the katago binary
  │  │     └─ build-eigen/      # CPU (Eigen) build; contains the katago binary
  │  └─ run/                    # Runtime folder: models (.bin.gz), configs, logs
  │     ├─ default_model.bin.gz # Symlink to the chosen neural net model
  │     ├─ default_gtp.cfg      # GTP config (edited after benchmarking)
  │     ├─ analysis.cfg         # JSON analysis engine config (copied from example)
  │     ├─ gtp_logs/            # Logs from GTP sessions
  │     └─ analysis_logs/       # Logs from analysis engine runs
  ├─ datago/                    # Our research workspace for RAG-augmented MCTS (new)
  └─ venv/                      # Python virtual environment for tooling

Quick start: run KataGo (GPU)

1) Activate this directory for runtime operations:
   cd /scratch2/f004h1v/alphago_project/katago_repo/run

2) Run the GTP engine (uses GPU/OpenCL build and the linked model):
   ../KataGo/cpp/build-opencl/katago gtp -model default_model.bin.gz -config default_gtp.cfg

   Notes:
   - Rules, visits, and thread parameters come from default_gtp.cfg.
   - First GPU run tuned kernels and saved tuning params under ~/.katago/opencltuning.

3) Run the JSON analysis engine (batch-friendly):
   ../KataGo/cpp/build-opencl/katago analysis -model default_model.bin.gz -config analysis.cfg

   - Produces JSON messages on stdin/stdout (see KataGo docs for message formats).
   - analysis.cfg is copied from cpp/configs/analysis_example.cfg and can be edited.

Files and folders in depth

katago_repo/KataGo (upstream source)
- README.md: Top-level documentation and usage overview.
- Compiling.md: Detailed build instructions for Linux/Mac/Windows and backends.
- docs/: Extended documentation (Analysis_Engine.md, GTP_Extensions.md, GraphSearch.md, KataGoMethods.md, rules.html, etc.).
- cpp/: C++ source and the build directories we created.
  - build-opencl/katago: GPU-enabled binary (OpenCL backend) – this is what we run.
  - build-eigen/katago: CPU-only binary (Eigen backend) – slower fallback or CI sanity.
  - configs/: Example configuration files you can copy and customize.
    - gtp_example.cfg: Example GTP config – we copied this to default_gtp.cfg.
    - analysis_example.cfg: Example for the JSON analysis engine – copied to analysis.cfg.
  - command/*: CLI entry points for commands like gtp, analysis, benchmark, match, etc.
  - neuralnet/*: Backends (OpenCL, CUDA, TensorRT, Eigen), kernels, and NN inference.
  - search/*: MCTS and search logic, helpers, analysis data.
  - game/*: Board, rules, history representations.
  - dataio/*: SGF, file IO, model loading, training IO.

katago_repo/run (our runtime folder)
- default_model.bin.gz (symlink): Points to the selected neural net.
  - Currently: kata1-b28c512nbt-s11653980416-d5514111622.bin.gz (latest kata1 b28c512 model at time of setup).
  - If you change the model file, update the symlink.
- default_gtp.cfg: GTP configuration used by the gtp command.
  - We set numSearchThreads = 24 based on benchmark results on NVIDIA RTX 6000 Ada (OpenCL).
  - Adjust maxVisits/time controls, rules, GPU selection, and logging here.
- analysis.cfg: Config for the JSON analysis engine.
  - Tune nnMaxBatchSize, numAnalysisThreads, and maxVisits as needed.
- gtp_logs/: Timestamped logs created when GTP runs (logDir configured in default_gtp.cfg).
- analysis_logs/: Timestamped logs created when analysis runs.

venv (python virtual environment)
- Location: /scratch2/f004h1v/alphago_project/venv
- Activate:
  source /scratch2/f004h1v/alphago_project/venv/bin/activate
- Currently includes numpy and requests for simple tooling/clients.
- For querying the analysis engine from Python, see KataGo/python/query_analysis_engine_example.py.

How configs and logs work
- default_gtp.cfg: Controls GTP behavior (search threads, visits/time, rules, resign, pondering, device selection, logging, etc.).
  - We derived it by copying cpp/configs/gtp_example.cfg to run/default_gtp.cfg and then editing.
  - You can set rules here (e.g., rules=japanese/chinese/tromp-taylor), resign thresholds, GPU device (openclGpuToUse), and logging (logDir).
- analysis.cfg: Based on cpp/configs/analysis_example.cfg. The analysis engine provides batched JSON input/output for backend services and tooling.
- Logs: GTP and analysis logs live under run/gtp_logs and run/analysis_logs respectively, controlled by logDir in configs.

Build artifacts (binaries)
- GPU/OpenCL binary:
  /scratch2/f004h1v/alphago_project/katago_repo/KataGo/cpp/build-opencl/katago
  - Built with: cmake .. -DUSE_BACKEND=OPENCL && make -j $(nproc)
  - Uses OpenCL 3.0 (NVIDIA CUDA platform) with FP16 tensor cores enabled where beneficial.
- CPU/Eigen binary:
  /scratch2/f004h1v/alphago_project/katago_repo/KataGo/cpp/build-eigen/katago
  - Built with: cmake .. -DUSE_BACKEND=EIGEN -DUSE_AVX2=1 && make -j $(nproc)
  - Slower; useful for environments without working GPUs.

Benchmarking and tuning
- Command (run folder):
  ../KataGo/cpp/build-opencl/katago benchmark -model default_model.bin.gz -config default_gtp.cfg
- What it does:
  - Autotunes OpenCL kernels on first run for the current GPU/model size; saves tuning under ~/.katago/opencltuning.
  - Sweeps numSearchThreads and recommends a value for your time control/visit budget.
  - We set numSearchThreads = 24 from these results.

Typical commands
- GTP engine:
  ../KataGo/cpp/build-opencl/katago gtp -model default_model.bin.gz -config default_gtp.cfg
- Analysis engine (JSON):
  ../KataGo/cpp/build-opencl/katago analysis -model default_model.bin.gz -config analysis.cfg
- Version:
  ../KataGo/cpp/build-opencl/katago version

What the model file is
- default_model.bin.gz points to a large neural network file (here: kata1-b28c512nbt-s11653980416-d5514111622.bin.gz).
- Models are downloaded from KataGo training: https://katagotraining.org/networks/ (served via https://media.katagotraining.org/...)
- You can swap to other size models (e.g., b18c384nbt for speed) if needed; update the symlink.

How we’ll work with this project
- Day-to-day runs happen inside katago_repo/run to keep configs and logs localized.
- Binary selection:
  - Prefer the GPU build (build-opencl/katago) on this machine.
  - CPU build is available for portability or quick sanity checks.
- Python tooling lives in the top-level venv for simple clients and later RAG indexing.

Next steps for the RAG project (datago)
- The datago folder will contain our RAG memory/index and integration tools per the plan.
- High-level components to add:
  1) Embeddings extraction: wrapper that asks the KataGo analysis engine for features or uses model trunk activations where available.
  2) ANN index (FAISS/HNSW): build a small curated memory (a few thousand entries) for fast root-only retrieval.
  3) Gating and blending: entropy-triggered prior blending at the root; optional tiny value nudge.
  4) Maintenance: add/prune cycle with importance scoring to keep memory fresh and compact.
- We will provide simple CLIs/notebooks to:
  - Populate the memory from SGFs/self-play.
  - Run head-to-head matches at fixed visits vs. baseline.
  - Log win rate, latency overhead, and ablations.

References
- KataGo GitHub repository: https://github.com/lightvector/KataGo
- KataGo networks: https://katagotraining.org/networks/


