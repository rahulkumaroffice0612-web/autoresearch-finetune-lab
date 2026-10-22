"""Load Typed Decisions cases from local JSONL, parsing the JSON-string columns.

The Hub stores `state`/`questions`/`gold`/`factors`/`label_agreement` as JSON
strings. We parse `gold` (needed for scoring) and keep the rest as parsed dicts.
`factors`/`label_agreement` are NOT model input (design doc §2) but we keep
`factors` available for the Arm B generator audit later.
"""
from __future__ import annotations

import json
from pathlib import Path

_JSON_COLS = ("state", "questions", "gold", "factors", "label_agreement")


def load_cases(path: str | Path) -> list[dict]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for col in _JSON_COLS:
                if col in row and isinstance(row[col], str):
                    row[col] = json.loads(row[col])
            cases.append(row)
    return cases


def write_cases(path: str | Path, cases: list[dict]) -> None:
    """Write cases back as JSONL, re-serialising the dict columns to JSON strings
    so the on-disk form is byte-identical to what we downloaded (stable hashing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for case in cases:
            row = dict(case)
            for col in _JSON_COLS:
                if col in row and not isinstance(row[col], str):
                    row[col] = json.dumps(row[col], ensure_ascii=False)
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
