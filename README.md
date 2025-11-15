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
datago – RAG-Augmented MCTS Workspace

Purpose
- Implement the selective Retrieval-Augmented (RAG) assistance around KataGo’s search, as outlined in the plan: entropy-gated root prior blending, optional leaf prior blending, and a compact curated memory with ANN.

Initial structure (proposed)
- notebooks/                  # Exploration/visual sanity checks
- src/
  - embeddings/               # State encoders / feature extraction wrappers
  - memory/                   # ANN index (FAISS/HNSW), add/prune, IO
  - gating/                   # Entropy and similarity gates
  - blend/                    # Prior/optional value blending utilities
  - clients/                  # JSON-analysis-engine clients and helpers
  - eval/                     # Match runners, metrics, ablations
- data/
  - memory/                   # Serialized memory shards/checkpoints
  - sgf/                      # Curated SGF positions for testing
  - tmp/                      # Scratch outputs, plots

Dependencies
- Python 3.12+ (use the project venv):
  source /scratch2/f004h1v/alphago_project/venv/bin/activate
- Add when implementing:
  pip install faiss-cpu or faiss-gpu  # per host setup
  pip install numpy pandas tqdm pyyaml rich

Interfaces to KataGo
- Preferred interface: JSON analysis engine.
  - Start from run/: ../KataGo/cpp/build-opencl/katago analysis -model default_model.bin.gz -config analysis.cfg
  - Use src/clients to: send-analyze-requests, parse JSON, and retrieve policy/value/ownership and (if available) feature vectors.
- Alternative: GTP with kata-analyze (less convenient for batch).

Key algorithms
- Entropy gating: compute H(policy) at root and compare to H_max.
- Retrieval: encode state -> ANN query -> top-K neighbors -> alignment (rot/ref) -> build retrieval prior.
- Blending: P'(a) = (1-β) P_nn(a) + β P_ret(a) at root; tiny value nudge optional.
- Maintenance: importance-scored add/prune; keep memory size M small (≈5k entries).

### Algorithm detail — two-step RAG augmentation (recommended)

- Entropy gating (both expansion and simulation):
  - At each decision point (root expansion and at selective intermediate nodes during simulation), compute the policy entropy H(P) = -∑_a P(a) log P(a). If H(P) exceeds a configured threshold H_trigger, trigger retrieval and augmentation for that step. This allows more aggressive augmentation only on uncertain/complex positions.
- Retrieval and reranking:
  - Query ANN for the top-K nearest stored embeddings for the current state.
  - Rerank these K candidates by a combination of:
    - Reachability score: estimate how reachable the candidate position is from the current node (e.g., short move-distance, high probability under the current policy, or a learned reachability model). Use a normalized score r ∈ [0,1].
    - Structural relation boost: boost parent or direct-child positions when a stored state is recognized as a direct ancestor/descendant (exact move match after un-rotation) — apply a multiplicative factor s > 1 when detected.
    - Combined ranking weight w = α * r + γ * s where α, γ are tunable.
  - The reranked neighbors produce an action prior P_nn(a) by aligning neighbor move(s) to the current board orientation and accumulating probability mass for their suggested moves.
- Expansion augmentation:
  - During node expansion when H(P) > H_trigger, construct a blended prior for the new child actions:
    - P_blend(a) = (1 - β_exp) * P_net(a) + β_exp * P_nn(a), where β_exp is the expansion blending weight.
  - If retrieved neighbors include parent/child states mapping to a single move, give that move an extra nudge proportional to the parent/child boost to reflect proven reachability.
- Simulation augmentation:
  - During simulation (rollouts or value/policy-guided playouts), if a simulation reaches a node whose entropy crosses H_trigger and ANN retrieval finds similar states, mix retrieved move distributions into the simulation policy at that node by:
    - P_sim(a) = (1 - β_sim) * P_current(a) + β_sim * P_nn(a)
  - This biases simulations toward moves seen in similar complex positions without fully overriding network guidance. β_sim may be smaller than β_exp.
- Storing positions during simulations:
  - If a simulation visits a node with high complexity (e.g., H(P) ≥ H_store and/or large disagreement between value estimate and rollout), capture a compact entry (embedding, canonicalized board, best-move candidates, metadata like frequency and timestamp) to a temporary buffer.
  - Buffer entries are periodically merged into the persistent ANN memory on low-load intervals or per configured batch size.
- Maintenance, pruning and lifecycle:
  - When the persistent memory size exceeds a configured population threshold M_max, trigger a pruning-and-add cycle:
    - Compute an importance score for each entry (recentness, frequency of retrieval, retrieval relevance, outcome-based utility) and evaluate its reachability-based ranking using the same rerank metric w = α * r + γ * s (reachability r and structural boost s).
    - Prune entries with low combined score (effectively those judged most "unreachable" under the rerank metric) — i.e., use the same reachability/unreachability metric you use to rerank retrieval results. After pruning, add buffered new entries or reweighted candidates.
  - Periodic re-indexing may be necessary for ANN structures (HNSW/FAISS) to keep performance predictable.

### Implementation notes / Contracts
- Inputs/outputs:
  - Embedding extractor signature: embed(state) -> float[d].
  - Retrieval signature: retrieve(embed, K) -> list[(id, score, canonical_state, action_hint, metadata)].
  - Rerank function: rerank(retrieved, state) -> weighted list used to build P_nn(a).
- Failure modes:
  - If ANN retrieval fails or returns low-quality matches (low cosine similarity), the system should gracefully fall back to network-only policy.
- Edge cases:
  - Transpositions and symmetries: canonicalize states consistently (rot/ref) to increase matches; align moves back to current orientation on blending.
  - Large branching factors: restrict blending to top-N moves per node to limit effect on search distribution.
  - Concurrency: only commit new entries after consensus or periodic batching to avoid high write contention on indices.

Milestones (from the 4-week plan)
1) Week 1: Baseline + tooling
   - Run baseline fixed-visit matches from analysis clients; export CSV of entropies/top-moves.
2) Week 2: Embeddings + ANN
   - Implement embedding extraction and HNSW index; build a small curated memory.
3) Week 3: Root-only RAG
   - Wire entropy gating + prior blending; online add/prune; run matches/ablations.
4) Week 4: Optional leaf blend + light tune
   - Small leaf blending; light encoder tune; finalize report.

How to start development here
1) Set up dependencies in venv.
2) Create src/clients to talk to analysis engine; write a minimal request/response loop.
3) Implement entropy estimation + logging for a test set of positions.
4) Stand up FAISS index with a tiny memory; build prior from neighbors.
5) Add a command-line tool to run a root-only RAG evaluation on SGF positions.

References
- KataGo GitHub: https://github.com/lightvector/KataGo
- Networks (models): https://katagotraining.org/networks/


