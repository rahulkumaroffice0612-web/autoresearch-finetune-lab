"""Day-1 data preparation: download, seal, and split. Run once.

Steps (design doc §11 Day 1, §19.1):
  1. Download typed-decisions `all` train+test from the Hub.
  2. Seal `test`: write data/sealed/test.jsonl (gitignored) + a tracked SHA-256
     manifest. The raw test must never be committed or read by the worker.
  3. Carve `train` -> fit / select / calib, stratified by workflow, deterministic
     seed. Write each split + a tracked SHA-256 manifest.

No model is touched here. Scoring the sealed test is a SEPARATE, declared step.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from data_io import load_cases, write_cases

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEALED = DATA / "sealed"
SPLITS = DATA / "splits"

SEED = 20261023
# Per design doc §19.1: 240 holdout cases (fraction≈0.2), split 120/120.
SELECT_PER_WF = 30
CALIB_PER_WF = 30


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(name: str, path: Path, cases: list[dict]) -> dict:
    entry = {
        "name": name,
        "file": str(path.relative_to(ROOT)),
        "n_cases": len(cases),
        "n_decisions": sum(len(c["gold"]) for c in cases),
        "sha256": sha256_file(path),
        "case_ids_sha256": hashlib.sha256(
            "\n".join(sorted(c["id"] for c in cases)).encode()
        ).hexdigest(),
    }
    return entry


def download() -> tuple[list[dict], list[dict]]:
    train = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    test = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    raw_train = DATA / "train.jsonl"
    raw_test = SEALED / "test.jsonl"
    # Persist through our own writer so hashes are stable and reproducible.
    write_cases(raw_train, [dict(r) for r in train])
    write_cases(raw_test, [dict(r) for r in test])
    return load_cases(raw_train), load_cases(raw_test)


def stratified_split(train: list[dict]) -> dict[str, list[dict]]:
    by_wf: dict[str, list[dict]] = defaultdict(list)
    for c in train:
        by_wf[c["workflow"]].append(c)

    rng = random.Random(SEED)
    fit, select, calib = [], [], []
    for wf in sorted(by_wf):
        cases = sorted(by_wf[wf], key=lambda c: c["id"])  # deterministic base order
        rng.shuffle(cases)
        select += cases[:SELECT_PER_WF]
        calib += cases[SELECT_PER_WF:SELECT_PER_WF + CALIB_PER_WF]
        fit += cases[SELECT_PER_WF + CALIB_PER_WF:]
    return {"fit": fit, "select": select, "calib": calib}


def main() -> None:
    SEALED.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)

    print("[prepare] downloading typed-decisions (all) ...", flush=True)
    train, test = download()
    print(f"[prepare] train={len(train)} test={len(test)}", flush=True)
    assert len(train) == 1200, f"expected 1200 train cases, got {len(train)}"
    assert len(test) == 400, f"expected 400 test cases, got {len(test)}"

    # --- seal test ---
    test_manifest = write_manifest("test", SEALED / "test.jsonl", test)
    (SEALED / "test.sha256").write_text(json.dumps(test_manifest, indent=2) + "\n")
    print(f"[seal] test sealed: {test_manifest['sha256'][:16]}... "
          f"({test_manifest['n_decisions']} decisions)", flush=True)

    # --- split train, stratified, frozen ---
    splits = stratified_split(train)
    manifests = {}
    for name, cases in splits.items():
        path = SPLITS / f"{name}.jsonl"
        write_cases(path, cases)
        manifests[name] = write_manifest(name, path, cases)
        print(f"[split] {name}: {len(cases)} cases / {manifests[name]['n_decisions']} decisions",
              flush=True)

    # integrity checks: no overlap, full coverage
    ids = {n: {c["id"] for c in cs} for n, cs in splits.items()}
    assert not (ids["fit"] & ids["select"]), "fit/select overlap"
    assert not (ids["fit"] & ids["calib"]), "fit/calib overlap"
    assert not (ids["select"] & ids["calib"]), "select/calib overlap"
    all_split_ids = ids["fit"] | ids["select"] | ids["calib"]
    assert all_split_ids == {c["id"] for c in train}, "splits do not cover train exactly"
    test_ids = {c["id"] for c in test}
    assert not (all_split_ids & test_ids), "train/test id collision"

    (SPLITS / "manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    print("[prepare] done. Manifests written; commit them to pre-register the splits.",
          flush=True)


if __name__ == "__main__":
    main()
