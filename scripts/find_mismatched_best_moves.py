#!/usr/bin/env python3
"""
Scan rag_data_o JSON files to find entries whose shallow best move differs from
the deep-analysis best move. Matching entries are written back out using the
original JSON structure under analysis/mismatched_best_moves/<gpu>/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


GPU_DIRS = ["gpu1", "gpu2", "gpu3", "gpu5", "gpu6"]
ROOT = Path(__file__).resolve().parents[1]  # RAGFlow-Datago/
SELFPLAY_ROOT = ROOT / "selfplay_output"
OUTPUT_ROOT = ROOT / "analysis" / "mismatched_best_moves"


def moves_differ(entry: Dict[str, Any]) -> bool:
    """Check if the shallow vs deep-analysis best moves disagree."""
    shallow = entry.get("best_move") or {}
    deep_result = entry.get("deep_result") or {}
    deep_best = deep_result.get("best_move") or {}
    shallow_move = shallow.get("move")
    deep_move = deep_best.get("move")
    return bool(shallow_move and deep_move and shallow_move != deep_move)


def filter_entry(data: Dict[str, Any]) -> Dict[str, Any] | None:
    """Return a filtered copy of the JSON dict containing only mismatched entries."""
    flagged: List[Dict[str, Any]] = data.get("flagged_positions") or []
    mismatches = [entry for entry in flagged if moves_differ(entry)]
    if not mismatches:
        return None

    filtered = dict(data)
    filtered["flagged_positions"] = mismatches

    summary = dict(filtered.get("summary") or {})
    total_moves = summary.get("total_moves")
    summary["flagged_count"] = len(mismatches)
    if isinstance(total_moves, (int, float)) and total_moves:
        summary["flagging_rate"] = len(mismatches) / total_moves
    else:
        summary["flagging_rate"] = None
    filtered["summary"] = summary
    return filtered


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for gpu in GPU_DIRS:
        data_dir = SELFPLAY_ROOT / gpu / "rag_data_o"
        if not data_dir.is_dir():
            continue

        for json_path in sorted(data_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text())
            except json.JSONDecodeError as exc:
                print(f"Failed to parse {json_path}: {exc}")
                continue

            filtered = filter_entry(data)
            if not filtered:
                continue

            out_path = OUTPUT_ROOT / gpu / json_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(filtered, indent=2))
            results.append((json_path, len(filtered["flagged_positions"])))

    if not results:
        print("No mismatched best-move entries found.")
        return

    print("Wrote mismatched entries:")
    for src, count in results:
        print(f"  {src} -> {count} entries")


if __name__ == "__main__":
    main()
