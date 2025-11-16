#!/usr/bin/env python3
"""
Copy rag_data_o JSON files into new rag_data directories while ensuring that
each best_move (shallow and deep) includes the child_sym_hash pulled from the
corresponding children list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


GPU_DIRS = ["gpu1", "gpu2", "gpu3", "gpu5", "gpu6"]
ROOT = Path(__file__).resolve().parents[1]  # RAGFlow-Datago/
SELFPLAY_ROOT = ROOT / "selfplay_output"


def build_child_map(children: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str | None], str]:
    """
    Build a lookup from (move, hash) -> child_sym_hash. The hash in the tuple is the
    child's own "hash" field if present, otherwise None. This lets us match by move
    first and fall back to hash disambiguation when available.
    """
    mapping: Dict[Tuple[str, str | None], str] = {}
    for child in children or []:
        move = child.get("move")
        child_hash = child.get("hash")
        child_sym = child.get("child_sym_hash")
        if not move or not child_sym:
            continue
        key = (move, child_hash if isinstance(child_hash, str) else None)
        if key not in mapping:
            mapping[key] = child_sym
        # also store fallback by move only
        mapping.setdefault((move, None), child_sym)
    return mapping


def attach_sym_hash(best_move: Dict[str, Any], mapping: Dict[Tuple[str, str | None], str]) -> bool:
    """
    Attach the child_sym_hash to best_move if possible. Returns True if updated.
    """
    if not best_move or "child_sym_hash" in best_move:
        return False

    move = best_move.get("move")
    if not isinstance(move, str):
        return False

    candidate_key = (move, best_move.get("hash") if isinstance(best_move.get("hash"), str) else None)
    child_sym = mapping.get(candidate_key) or mapping.get((move, None))
    if not child_sym:
        return False

    best_move["child_sym_hash"] = child_sym
    return True


def process_file(src_path: Path, dest_path: Path) -> Tuple[int, int]:
    """Return tuple (#shallow updates, #deep updates)."""
    data = json.loads(src_path.read_text())
    shallow_updates = 0
    deep_updates = 0

    for pos in data.get("flagged_positions") or []:
        child_map = build_child_map(pos.get("children") or [])
        if pos.get("best_move") and attach_sym_hash(pos["best_move"], child_map):
            shallow_updates += 1

        deep = pos.get("deep_result") or {}
        deep_children = deep.get("children") or []
        deep_map = build_child_map(deep_children)
        if deep.get("best_move") and attach_sym_hash(deep["best_move"], deep_map):
            deep_updates += 1

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(json.dumps(data, indent=2))
    return shallow_updates, deep_updates


def main() -> None:
    for gpu in GPU_DIRS:
        src_dir = SELFPLAY_ROOT / gpu / "rag_data_o"
        dst_dir = SELFPLAY_ROOT / gpu / "rag_data"
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.is_dir():
            print(f"[WARN] Missing source directory: {src_dir}")
            continue

        for src_file in sorted(src_dir.glob("*.json")):
            dst_file = dst_dir / src_file.name
            shallow, deep = process_file(src_file, dst_file)
            print(f"{src_file} -> shallow:{shallow} deep:{deep}")


if __name__ == "__main__":
    main()
