from __future__ import annotations

import json
from pathlib import Path


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in str(text or "").replace(";", " ").replace(",", " ").split() if tok.strip()}


class MemoryStore:
    """Small JSONL success/correction memory used for artifact demos."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.success_path = self.root / "success_kb.jsonl"
        self.correction_path = self.root / "correction_kb.jsonl"

    def _read(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _append(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def retrieve(self, query_text: str, top_k: int = 3, include_corrections: bool = True) -> list[dict]:
        rows = list(self._read(self.success_path))
        if include_corrections:
            rows.extend(self._read(self.correction_path))
        query_tokens = _tokens(query_text)
        ranked = sorted(
            rows,
            key=lambda row: len(query_tokens & _tokens(row.get("search_text") or row.get("experience"))),
            reverse=True,
        )
        return ranked[: max(0, int(top_k))]

    def write(self, kind: str, row: dict) -> None:
        path = self.success_path if kind == "success" else self.correction_path
        self._append(path, row)

