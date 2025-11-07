"""
scripts/run_rag_harness.py

Simple harness to demonstrate the non-invasive RAG prototype flow:
- load an example KataGo analysis JSON
- create an embedding
- query the ANN memory (if present)
- rerank and build retrieval prior
- blend with a fake network prior and print results

This is a smoke-test example to show how the scaffolds connect.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import os
import sys

# local imports
from datago.src.clients.katago_client import analysis_to_embedding, EmbeddingProjector
from datago.src.memory.index import ANNIndex
from datago.src.blend.blend import rerank_neighbors, build_retrieval_prior, blend_priors


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def fake_network_prior(policy_vec: np.ndarray) -> dict:
    # build a fake mapping of move_ix->prob for top moves
    probs = policy_vec / (policy_vec.sum() + 1e-12)
    top_k = min(16, len(probs))
    idxs = np.argsort(-probs)[:top_k]
    return {str(int(i)): float(probs[i]) for i in idxs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("analysis_json", help="path to KataGo analysis JSON sample")
    p.add_argument("--memory", help="path to memory dir (optional)")
    args = p.parse_args()

    j = load_json(args.analysis_json)
    proj = EmbeddingProjector(d_out=128)
    emb, ar = analysis_to_embedding(j, projector=proj)

    # Load memory if provided
    ann = ANNIndex(dim=128)
    if args.memory and os.path.exists(args.memory):
        ann.load(args.memory)
        print("Loaded memory entries:", len(ann))
    else:
        print("No memory loaded; running with empty memory.")

    neighbors = ann.retrieve(emb, k=8) if len(ann) > 0 else []
    reranked = rerank_neighbors(neighbors, ar.policy)
    p_nn = build_retrieval_prior(reranked)
    p_net = fake_network_prior(ar.policy)
    p_blend = blend_priors(p_net, p_nn, beta=0.4)

    print("Network prior (top)", sorted(p_net.items(), key=lambda x: -x[1])[:8])
    print("Retrieval prior (top)", sorted(p_nn.items(), key=lambda x: -x[1])[:8])
    print("Blended prior (top)", sorted(p_blend.items(), key=lambda x: -x[1])[:8])


if __name__ == "__main__":
    main()
