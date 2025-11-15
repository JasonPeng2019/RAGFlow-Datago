"""
scripts/populate_memory.py

Create a small memory snapshot from the sample analysis JSON. This will create several MemoryEntry
objects with slight variations and save an ANNIndex under the specified memory directory.

Usage:
  PYTHONPATH=. python3 datago/scripts/populate_memory.py --out data/memory/sample_mem
"""
from __future__ import annotations

import argparse
import os
import json
import numpy as np

from datago.src.clients.katago_client import analysis_to_embedding, EmbeddingProjector
from datago.src.memory.index import ANNIndex
from datago.src.memory.schema import MemoryEntry


def top_moves_from_policy(policy: np.ndarray, topk: int = 3):
    probs = policy / (policy.sum() + 1e-12)
    idxs = np.argsort(-probs)[:topk]
    return [{"move": str(int(i)), "prob": float(probs[i])} for i in idxs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default="datago/tests/sample_analysis.json")
    p.add_argument("--out", default="data/memory/sample_mem")
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    with open(args.sample, "r") as f:
        j = json.load(f)

    proj = EmbeddingProjector(d_out=128)

    # Use base embedding then perturb to create several entries
    base_emb, ar = analysis_to_embedding(j, projector=proj)
    # canonicalize policy for storage
    try:
        canon_policy, canon_sym = proj and None or (None, None)
    except Exception:
        canon_policy, canon_sym = None, 0
    ann = ANNIndex(dim=128)

    os.makedirs(args.out, exist_ok=True)

    for i in range(args.n):
        # small perturbation
        noise = np.random.RandomState(i).normal(scale=0.01, size=base_emb.shape)
        emb = (base_emb + noise).astype(np.float32)
        # compute canonical policy and store symmetry info on the first iteration
        if i == 0:
            try:
                from datago.src.clients.katago_client import canonicalize_policy
                cpol, csym = canonicalize_policy(ar.policy, board_x=19, board_y=19)
            except Exception:
                cpol, csym = ar.policy, 0
        # perturb the canonical policy slightly for diversity
        use_policy = (cpol + np.random.RandomState(i).normal(scale=0.001, size=cpol.shape)).clip(min=0.0)
        best_moves = top_moves_from_policy(use_policy, topk=4)
        entry = MemoryEntry.create(embed=emb, canonical_board=f"sample_board_{i}", best_moves=best_moves)
    # mark board size and stored symmetry (we store entries in symmetry 0 for this prototype)
        entry.metadata["board_x"] = 19
        entry.metadata["board_y"] = 19
        entry.metadata["symmetry"] = int(csym if 'csym' in locals() else 0)
        # add some metadata hints: mark the first entry as parent relation
        if i == 0:
            entry.metadata["relation"] = "parent"
            entry.metadata["structural_boost"] = 2.0
        ann.add(entry)

    ann.save(args.out)
    print("Saved memory to", args.out)


if __name__ == "__main__":
    main()
