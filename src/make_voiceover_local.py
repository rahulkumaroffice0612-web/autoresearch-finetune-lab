#!/usr/bin/env python3
"""Free, offline demo voiceover using macOS `say` (no account, no cost).

A drop-in alternative to the ElevenLabs path (src/make_voiceover.mjs) for when
upgrading the ElevenLabs account isn't an option. Parses the same narration file
(artifacts/demo_voiceover.md), synthesizes each `## Scene N` block with the
built-in macOS speech engine, converts AIFF -> MP3 with ffmpeg, and writes a
concatenated narration.mp3.

Requires: macOS `say` (built in) and `ffmpeg` (brew install ffmpeg).

Usage:
  .venv/bin/python src/make_voiceover_local.py                 # all scenes
  .venv/bin/python src/make_voiceover_local.py --voice Daniel  # pick a voice
  .venv/bin/python src/make_voiceover_local.py --scene 3       # one scene
  .venv/bin/python src/make_voiceover_local.py --rate 170      # words/min

Tip: list installed voices with `say -v '?'`. For a more natural voice, install
a "Premium/Enhanced" one in System Settings -> Accessibility -> Spoken Content
-> System Voice -> Manage Voices, then pass it with --voice.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICEOVER_MD = ROOT / "artifacts" / "demo_voiceover.md"
OUT_DIR = ROOT / "artifacts" / "demo_audio"


def require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        sys.exit(f"'{tool}' not found. This generator needs macOS `say` and `ffmpeg`.")
    return path


def parse_scenes(md: str) -> list[tuple[int, str]]:
    parts = re.split(r"^## Scene (\d+).*$", md, flags=re.MULTILINE)
    scenes: list[tuple[int, str]] = []
    for i in range(1, len(parts), 2):
        num = int(parts[i])
        body = parts[i + 1]
        quoted = [ln.lstrip()[1:].strip() for ln in body.splitlines() if ln.lstrip().startswith(">")]
        text = " ".join(q for q in quoted if q)
        if text:
            scenes.append((num, text))
    return scenes


def synth_scene(text: str, voice: str, rate: int, mp3_path: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        aiff = Path(tmp.name)
    try:
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text], check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
             "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path)],
            check=True,
        )
    finally:
        aiff.unlink(missing_ok=True)


def concat_mp3(parts: list[Path], out: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lst:
        for p in parts:
            lst.write(f"file '{p.resolve()}'\n")
        list_path = Path(lst.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(list_path), "-codec:a", "libmp3lame", "-q:a", "2", str(out)],
            check=True,
        )
    finally:
        list_path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Samantha", help="macOS voice name (see `say -v '?'`)")
    ap.add_argument("--rate", type=int, default=165, help="words per minute")
    ap.add_argument("--scene", type=int, default=None, help="only this scene number")
    args = ap.parse_args()

    require("say")
    require("ffmpeg")

    scenes = parse_scenes(VOICEOVER_MD.read_text())
    if args.scene is not None:
        scenes = [s for s in scenes if s[0] == args.scene]
        if not scenes:
            sys.exit(f"scene {args.scene} not found")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    for num, text in scenes:
        out = OUT_DIR / f"scene_{num}.mp3"
        synth_scene(text, args.voice, args.rate, out)
        made.append(out)
        print(f"wrote {out}  <- {text[:60]}...")

    if args.scene is None and made:
        full = OUT_DIR / "narration.mp3"
        concat_mp3(made, full)
        print(f"wrote {full}  (all {len(made)} scenes concatenated, voice={args.voice})")


if __name__ == "__main__":
    main()
