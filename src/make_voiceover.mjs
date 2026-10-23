// Generate the LazyDecider demo voiceover with the official ElevenLabs SDK.
//
// Reads narration from artifacts/demo_voiceover.md (one block per `## Scene N`),
// synthesizes each scene to artifacts/demo_audio/scene_N.mp3, and writes a
// concatenated narration.mp3. The API key is read from the repo-root .env
// (ELEVENLABS_API_KEY) — never committed.
//
// Usage:
//   npm install
//   node src/make_voiceover.mjs                 # all scenes
//   node src/make_voiceover.mjs --scene 3       # one scene
//   node src/make_voiceover.mjs --voice <id>    # override the voice
//
// Free-tier note: free ElevenLabs API keys cannot synthesize library voices
// (HTTP 402). Add $5 of API credits or use a personal/cloned voice.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";
import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const ENV_PATH = path.resolve(ROOT, "..", ".env");
const VOICEOVER_MD = path.join(ROOT, "artifacts", "demo_voiceover.md");
const OUT_DIR = path.join(ROOT, "artifacts", "demo_audio");

// A personal/owned voice on this account (override with --voice). Matches the
// id supplied for this project.
const DEFAULT_VOICE = "kLhAstPcnnPxqzk6gS5i";
const MODEL_ID = "eleven_multilingual_v2";

function loadKey() {
  const text = fs.readFileSync(ENV_PATH, "utf-8");
  for (const line of text.split("\n")) {
    if (line.startsWith("ELEVENLABS_API_KEY")) {
      return line.split("=").slice(1).join("=").trim().replace(/^["']|["']$/g, "");
    }
  }
  throw new Error(`ELEVENLABS_API_KEY not found in ${ENV_PATH}`);
}

function parseScenes(md) {
  // Split on `## Scene N ...` headings, capturing the number.
  const parts = md.split(/^## Scene (\d+).*$/m);
  const scenes = [];
  for (let i = 1; i < parts.length; i += 2) {
    const num = parseInt(parts[i], 10);
    const body = parts[i + 1] ?? "";
    // Spoken text = the blockquote lines (`> ...`).
    const text = body
      .split("\n")
      .filter((l) => l.trimStart().startsWith(">"))
      .map((l) => l.trimStart().slice(1).trim())
      .filter(Boolean)
      .join(" ");
    if (text) scenes.push({ num, text });
  }
  return scenes;
}

async function streamToBuffer(response) {
  // The SDK returns a web ReadableStream of audio bytes.
  const chunks = [];
  if (response && typeof response.getReader === "function") {
    const reader = response.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(Buffer.from(value));
    }
    return Buffer.concat(chunks);
  }
  // Fallback: async iterable (Node Readable).
  for await (const chunk of response) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks);
}

function parseArgs() {
  const args = process.argv.slice(2);
  const out = { voice: DEFAULT_VOICE, scene: null };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--voice") out.voice = args[++i];
    else if (args[i] === "--scene") out.scene = parseInt(args[++i], 10);
  }
  return out;
}

async function main() {
  const { voice, scene } = parseArgs();
  const client = new ElevenLabsClient({ apiKey: loadKey() });

  let scenes = parseScenes(fs.readFileSync(VOICEOVER_MD, "utf-8"));
  if (scene !== null) {
    scenes = scenes.filter((s) => s.num === scene);
    if (scenes.length === 0) throw new Error(`scene ${scene} not found`);
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  const combined = [];

  for (const { num, text } of scenes) {
    let audio;
    try {
      const response = await client.textToSpeech.convert(voice, {
        text,
        modelId: MODEL_ID,
      });
      audio = await streamToBuffer(response);
    } catch (err) {
      const msg = String(err?.body?.detail?.message ?? err?.message ?? err);
      if (String(err?.statusCode) === "402" || /library voice|paid_plan/i.test(msg)) {
        console.error(
          "\nElevenLabs HTTP 402 — this free key can't use this voice via the API.\n" +
            "Fix: add $5 of API credits, OR use a personal/cloned voice with --voice <id>.\n" +
            `Server said: ${msg}\n`
        );
        process.exit(1);
      }
      throw err;
    }
    const out = path.join(OUT_DIR, `scene_${num}.mp3`);
    fs.writeFileSync(out, audio);
    combined.push(audio);
    console.log(`wrote ${out}  (${Math.round(audio.length / 1024)} KB)  <- ${text.slice(0, 60)}...`);
  }

  if (scene === null && combined.length) {
    const full = path.join(OUT_DIR, "narration.mp3");
    fs.writeFileSync(full, Buffer.concat(combined));
    console.log(`wrote ${full}  (all ${combined.length} scenes concatenated)`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
