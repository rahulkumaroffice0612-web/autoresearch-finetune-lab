#!/usr/bin/env python3
"""Generate the demo voiceover with ElevenLabs text-to-speech.

Reads the narration from artifacts/demo_voiceover.md (one block per `## Scene N`
heading), synthesizes each scene to artifacts/demo_audio/scene_N.mp3, and also
writes a single concatenated narration.mp3.

Key: ELEVENLABS_API_KEY from the repo-root .env (never committed).

Usage:
  .venv/bin/python src/make_voiceover.py                 # all scenes
  .venv/bin/python src/make_voiceover.py --voice <id>    # pick a voice you own
  .venv/bin/python src/make_voiceover.py --scene 3       # one scene only

Notes on the free tier:
  A free ElevenLabs API key cannot synthesize the premade "library" voices
  (Rachel/Aria/etc.) — the API returns HTTP 402. To make this work either
  (a) add $5 of API credits, or (b) create/clone a personal voice in the
  ElevenLabs app and pass its id with --voice. This script surfaces that error
  clearly instead of writing a broken file.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_PARENT = ROOT.parent
VOICEOVER_MD = ROOT / "artifacts" / "demo_voiceover.md"
OUT_DIR = ROOT / "artifacts" / "demo_audio"

# Default voice: ElevenLabs "Rachel" (works once the account can use library
# voices). Override with --voice to use a personal/cloned voice on the free tier.
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"
MODEL_ID = "eleven_multilingual_v2"
API = "https://api.elevenlabs.io/v1/text-to-speech"


def load_key() -> str:
    env = REPO_PARENT / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("ELEVENLABS_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("ELEVENLABS_API_KEY not found in ../.env")


def parse_scenes(md: str) -> list[tuple[int, str]]:
    """Return [(scene_number, spoken_text), ...] from the `## Scene N` blocks."""
    scenes: list[tuple[int, str]] = []
    # Split on scene headings; capture the number.
    parts = re.split(r"^## Scene (\d+).*$", md, flags=re.MULTILINE)
    # parts = [preamble, num, body, num, body, ...]
    for i in range(1, len(parts), 2):
        num = int(parts[i])
        body = parts[i + 1]
        # Spoken text lives in the blockquote lines (`> ...`).
        quoted = [ln[1:].strip() for ln in body.splitlines() if ln.lstrip().startswith(">")]
        text = " ".join(q for q in quoted if q)
        if text:
            scenes.append((num, text))
    return scenes


def synth(text: str, voice: str, key: str) -> bytes:
    body = (
        '{"text": %s, "model_id": "%s", '
        '"voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.1}}'
    ) % (_json_str(text), MODEL_ID)
    req = urllib.request.Request(
        f"{API}/{voice}",
        data=body.encode("utf-8"),
        headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        if e.code == 402:
            sys.exit(
                "\nElevenLabs HTTP 402 — this free key can't use this voice via the API.\n"
                "Fix: add $5 of API credits, OR create a personal voice in the ElevenLabs app\n"
                "and re-run with --voice <your_voice_id>.\n"
                f"Server said: {detail}\n"
            )
        sys.exit(f"ElevenLabs HTTP {e.code}: {detail}")


def _json_str(s: str) -> str:
    import json
    return json.dumps(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="ElevenLabs voice id")
    ap.add_argument("--scene", type=int, default=None, help="only this scene number")
    args = ap.parse_args()

    key = load_key()
    scenes = parse_scenes(VOICEOVER_MD.read_text())
    if args.scene is not None:
        scenes = [s for s in scenes if s[0] == args.scene]
        if not scenes:
            sys.exit(f"scene {args.scene} not found")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = bytearray()
    for num, text in scenes:
        audio = synth(text, args.voice, key)
        out = OUT_DIR / f"scene_{num}.mp3"
        out.write_bytes(audio)
        combined += audio
        print(f"wrote {out}  ({len(audio)//1024} KB)  \u2190 {text[:60]}...")

    if args.scene is None and combined:
        full = OUT_DIR / "narration.mp3"
        full.write_bytes(bytes(combined))
        print(f"wrote {full}  ({len(combined)//1024} KB, all scenes concatenated)")


if __name__ == "__main__":
    main()
