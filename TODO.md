# datago Implementation TODO

This document collects the implementation plan, design details, parameter defaults, and actionable tasks for integrating a retrieval-augmented (RAG) vector memory with KataGo's MCTS and CNNs. It consolidates the design discussed previously: entropy-gated retrieval, k-NN + rerank by reachability/parent-child structural boost, blending for expansion and simulation, storing complex simulation states, and pruning by unreachability.

## Goals / high level
- Use entropy of the policy distribution to gate retrieval and augmentation at expansion and (selective) simulation nodes.
- Retrieve K nearest neighbors (ANN) for a position embedding, rerank neighbors by a reachability-based metric and a structural parent/child boost.
- Use reranked neighbors to build a retrieval prior P_nn(a) aligned to the current orientation, then blend with network priors during expansion and simulation.
- During simulations, capture complex/high-entropy positions into a buffer and add them to persistent memory in batches.
- When memory population exceeds M_max, run a prune/add cycle using the same reachability-based rerank metric so that "unreachable" states are pruned.

---

## Recommended approach
1. Prototype non-invasively using KataGo's JSON analysis engine. Use the policy logits/value to build embeddings and iterate quickly in Python.
2. After validating benefit, optionally integrate into KataGo C++ for lower-latency, production use (expose internal CNN activations or embed FAISS in C++ or call a local retrieval service via IPC).

---

## Actionable steps (ordered)

1) Decide integration approach (quick)
- Choose non-invasive prototype first (JSON analysis outputs). Document the eventual path to C++ integration if needed.

2) Prototype embedding extraction (implement first)
- File: `src/clients/katago_client.py` (or similar).
- Tasks:
  - Parse KataGo JSON analysis outputs (policy and value) from `katago analysis` or JSON analysis engine.
  - Canonicalize board orientation (rotate/reflect to canonical form) before embedding.
  - Construct embedding options:
    - Simple: normalized policy vector (size up to 19x19 = 361), plus value scalar.
    - Improved: logits if available; optional small learned MLP projection to reduce dim.
  - Add utilities to map neighbor moves (canonical orientation) back to current orientation.
- Test: run on a small SGF dataset, generate embeddings for a few hundred positions.

3) Build ANN index and persistent storage (`src/memory`)
- Choose implementation: FAISS (fast, mature) or hnswlib (simple HNSW interface). Both are fine; FAISS recommended if you expect GPU later.
- Design entry schema (per entry metadata):
  - id: uuid
  - embed: float[d]
  - canonical_state: compact board fingerprint
  - best_moves: list[(move, prob)]
  - metadata: {visits, last_seen_ts, importance, outcome_stats}
- Implement:
  - in-memory index wrapper (add, retrieve, remove)
  - persistent snapshot/save/load of index and metadata under `data/memory/`
  - shard format for large memories
- Smoke test: add ~500 curated positions from `data/sgf`, query and inspect returned neighbors.

4) Implement gating & retrieval (`src/gating` + `src/blend`)
- Gate: compute normalized entropy H_norm = H(P)/ln(#legal_moves)
  - H_trigger default: 0.7
  - H_store default: 0.9
- Retrieval interface: `retrieve(embed, K=16) -> list[Neighbor]` where Neighbor contains id, score, canonical_state, best_moves, metadata.
- Reranking:
  - For each neighbor compute:
    - reachability r ∈ [0,1]
    - structural boost s (e.g., s=2.0 if parent/child, else s=1.0)
  - Combined weight w = α * r + γ * s (recommended α=0.7, γ=0.3). Normalize w across neighbors.
  - Map neighbor best_moves into current orientation and aggregate to form retrieval prior P_nn(a).
- Blending:
  - Expansion: P_blend = normalize((1 - β_exp) * P_net + β_exp * P_nn)  (β_exp default = 0.4)
  - Simulation: P_sim = normalize((1 - β_sim) * P_current + β_sim * P_nn) (β_sim default = 0.15)
  - Limit blending to top-N moves (N=16) to keep effect focused.

5) Reachability estimator (`src/memory/reachability.py`)
- Options (increasing cost/accuracy):
  A) Parent/child exact check and move-distance heuristic (cheap).
  B) Policy-vector similarity (cosine) between current and candidate (cheap-medium).
  C) Short policy-guided rollouts to check empirical reachability (expensive; use for top candidates only).
  D) Learned model trained to predict reach-in-D moves (offline data required).
- Practical starting approach: B + A (policy similarity + parent/child boost). Later, for top-k candidates, run short rollouts to refine r.
- API: `estimate_reachability(current_state, candidate_state, budget=None) -> float`.

6) Expansion augmentation (hook points)
- Non-invasive prototype options:
  - Implement a custom MCTS harness that uses the KataGo network for evaluation but runs its own MCTS loop (allows blending at expansion/simulation without C++ changes).
  - Or orchestrate approximate expansion by re-simulating: evaluate root, if H > threshold, retrieve and inject blends into next decisions in harness.
- C++ integration:
  - Add a call in the expansion code path (search/search.cpp or equivalent) to request P_nn and blend with node priors before pushing children.
  - Retrieval/ANN options:
    - In-process: embed FAISS or hnswlib and the memory in C++.
    - Out-of-process: create a low-latency local RPC (unix domain socket / gRPC) service written in Python/C++ to handle retrieve/rerank.
- Pseudocode (conceptual):
```
if entropy(node.policy) > H_trigger:
    embed = extract_embedding(node)
    neighbors = memory.retrieve(embed, K)
    reranked = rerank(neighbors, node)
    P_nn = build_prior(reranked, node)
    node.prior = blend(node.prior, P_nn, beta_exp)
```

7) Simulation augmentation and storing
- During simulation, at each node if H > H_trigger (or other complexity metric), do light retrieval and blend with β_sim.
- Storing rule: if H >= H_store or |value_net - rollout_value| > disagreement_threshold (e.g., 0.2), capture a compact entry:
  - (embed, canonical_state, best_move_hint, metadata)
  - Append to a buffer.
- Buffer flush: on reaching BATCH_SIZE (e.g., 128) or low CPU load, add to persistent memory and update index.
- Ensure deduplication on insertion (by canonical_state fingerprint).

8) Maintenance, pruning and lifecycle
- When memory size > M_max (e.g., 5000), trigger prune/add:
  - For each entry compute importance score (recentness, retrieval frequency, utility), and reachability score using the same rerank metric (w = α * r + γ * s).
  - Combined score = λ_imp * importance + λ_reach * avg_reachability
  - Prune bottom p_prune (e.g., 15%) which will be the "most unreachable" according to the rerank metric.
  - Insert buffered new entries, reindex, and snapshot shards.
- Reindexing: schedule offline or staggered to avoid search latency spikes.

9) Tests, metrics, ablations
- Unit tests:
  - embedding canonicalization and mapping
  - retrieval-to-action mapping and orientation alignment
  - rerank weight math and pruning selection
  - reachability estimator behavior on synthetic pairs
- Experiments:
  - Baseline KataGo vs Root-only RAG vs Expansion+Simulation RAG
  - Metrics: winrate, avg entropy distribution, retrieval influence rate, query latency
- Ablations: vary β_exp, β_sim, H_trigger, reachability method, K.

10) Performance considerations
- Gate aggressively to limit retrieval calls (entropy thresholding, max retrievals per search).
- Cache retrieval responses per state fingerprint.
- Use local in-process index for production; use a fast local RPC for experimentation.
- For heavy reachability checks, restrict to top few neighbors only.

---

## Suggested file scaffolding (starter)
- `src/clients/katago_client.py`  -- handles JSON analysis, embedding extraction, canonicalization
- `src/memory/index.py`          -- ANN wrapper, add/retrieve/remove, persistence
- `src/memory/schema.py`         -- entry schema and metadata handling
- `src/memory/reachability.py`   -- reachability estimation utilities
- `src/gating/gate.py`           -- entropy gates, thresholds
- `src/blend/blend.py`           -- rerank, prior construction, blending functions
- `scripts/run_rag_harness.py`   -- prototype harness to run matches or tests using the RAG augmentation (non-invasive)

---

## Parameter defaults (initial)
- embedding_dim = 128
- K = 16
- β_exp = 0.4
- β_sim = 0.15
- H_trigger = 0.7 (normalized entropy)
- H_store = 0.9
- M_max = 5000
- p_prune = 0.15
- reachability weights: α = 0.7, γ = 0.3

---

## Data formats
- Memory entry JSON example:
```
{
  "id": "uuid4",
  "embed": [0.012, -0.34, ...],
  "canonical_board": "<19x19 string>",
  "best_moves": [{"move":"D4","prob":0.45}],
  "visits":120,
  "last_seen":169xxx,
  "winrate":0.41,
  "importance":0.72
}
```
- Keep FAISS/HNSW index file alongside a metadata mapping file for indices -> entry metadata.

---

## Pseudocode snippets
- Expansion blending
```
if entropy(P) > H_trigger:
    neighbors = memory.retrieve(embed, K)
    reranked = rerank(neighbors, current_state)
    P_nn = map_neighbors_to_actions(reranked)
    P_blend = normalize((1 - beta_exp) * P_net + beta_exp * P_nn)
    use P_blend for expansion priors
```

- Simulation storing
```
if entropy(P) >= H_store or abs(value_net - rollout_value) > disagreement_thresh:
    buffer.append({embed, canonical_state, action_hint, metadata})
if len(buffer) >= BATCH_SIZE:
    memory.batch_add(buffer)
    buffer.clear()
```

---

## Next steps you can ask me to do
- Scaffold the Python modules and small unit tests (I can create the files and run quick smoke checks).
- Produce concrete C++ hook locations in the KataGo repo (I can search the repo and point to specific files and functions to modify).
- Create a minimal local retrieval service (Python) and example client usage from C++ via a small IPC/gRPC example.

---

Prepared by: datago design notes

(If you want this in a different filename or to store in a tracker, tell me where and I'll move or duplicate it.)
