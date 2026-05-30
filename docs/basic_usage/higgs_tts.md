# Higgs TTS Model Usage

This guide uses [`boson-sglang/higgs-audio-v3-tts-4b-base`](https://huggingface.co/boson-sglang/higgs-audio-v3-tts-4b-base) — Higgs Audio v3 (Qwen3-4B backbone, 8 discrete codebooks × 1026 vocab, bf16) — with SGLang-Omni and the OpenAI-compatible API. The pipeline is `preprocessing → audio_encoder → tts_engine → vocoder`.

## Prerequisites

Build the runtime image from the repo's Dockerfile. This is the recommended,
reproducible path — it starts from the tuned CUDA / SGLang / FlashInfer base and
installs `sglang-omni` on top **without** reinstalling the base's SGLang /
FlashInfer build (reinstalling them from PyPI would replace the tuned build).
The image's entrypoint is `sgl-omni`.

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
docker build -f docker/Dockerfile.higgs-runtime -t sglang-omni:higgs .
```

The Higgs TTS model is private; download it on the host with your HF token (the
cache is mounted into the container at launch):

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
hf download boson-sglang/higgs-audio-v3-tts-4b-base
```

> **Installing into an existing environment instead of building the image?** Use
> the same exclude list the Dockerfile uses, so you don't overwrite the tuned
> SGLang / FlashInfer stack:
> ```bash
> printf '%s\n' sglang torch torchvision flashinfer flashinfer-python \
>   flashinfer-jit-cache > /tmp/excludes.txt
> uv pip install -v -e . --excludes /tmp/excludes.txt
> ```

## Launch the Server

```bash
docker run --gpus all --shm-size 32g -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  sglang-omni:higgs serve \
  --model-path boson-sglang/higgs-audio-v3-tts-4b-base \
  --config examples/configs/higgs_tts.yaml \
  --host 0.0.0.0 --port 8000
```

`serve` is passed to the image's `sgl-omni` entrypoint, and `--config` resolves
against the image's working directory (`/workspace/sglang-omni`). The audio codec
is bundled in the TTS checkpoint and loads automatically from `--model-path`; it
can't be swapped for a different codec.

## Use Curl

### Voice Cloning

Higgs TTS conditions on both reference audio **and** its transcript (`<|ref_text|>` segment); supplying the transcript materially improves quality versus audio-only cloning. The `references` field accepts `audio_path` (local path or HTTP URL) and `text` (transcript of that audio).

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

### Zero-shot

Without a reference, the model falls back to the `<|tts|> <|text|> ... <|audio|>` zero-shot prompt:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav
```

### Inline (base64) Reference Audio

Besides `references[].audio_path` (local path or HTTP URL), reference audio can be
passed inline with the top-level `ref_audio` field — useful when the client holds
the audio in memory and there is no shared filesystem or URL. `ref_audio` accepts
a `data:` URI, a raw base64 string, an `http(s)://` / `file://` URL, or a local
path; pair it with `ref_text` for the transcript. This works on both the HTTP and
WebSocket (`session.config`) paths.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "data:audio/wav;base64,UklGRiQAAABXQVZF...",
    "ref_text": "We asked over twenty different people, and they all said it was his."
  }' \
  --output output.wav
```

### Streaming

Set `"stream": true` to receive audio chunks over Server-Sent Events (SSE).
For low-latency playback, request raw 16-bit PCM chunks with
`"response_format": "pcm"`:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "response_format": "pcm",
    "stream": true
  }'
```

Each SSE event contains an `audio.speech.chunk` object. The audio bytes are
base64 encoded in `audio.data`; for PCM, `audio.format` is `pcm`,
`audio.mime_type` is `audio/pcm`, and `audio.sample_rate` is included in the
event. The stream ends with `data: [DONE]`.

The streaming chunk policy is configured server-side by the Higgs TTS pipeline.
Clients should not pass model-internal chunk sizing parameters for normal use.

### Streaming Text Input

Use the WebSocket endpoint when text arrives incrementally (e.g. token-by-token
from an upstream LLM) and the server should start synthesizing completed
sentences before the full text is available:

```python
import asyncio
import json
import websockets


async def main():
    async with websockets.connect(
        "ws://localhost:8000/v1/audio/speech/stream"
    ) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.config",
                    "response_format": "pcm",
                    "stream_audio": True,
                    "fastout": True,  # release the first chunk early (see below)
                }
            )
        )
        await ws.send(json.dumps({"type": "input.text", "text": "Hello world. "}))
        await ws.send(json.dumps({"type": "input.text", "text": "How are you?"}))
        await ws.send(json.dumps({"type": "input.done"}))

        async for message in ws:
            if isinstance(message, bytes):
                # Raw int16 PCM bytes when stream_audio=true.
                continue
            print(message)


asyncio.run(main())
```

**Client → server messages**

| Message | Purpose |
|---|---|
| `{"type": "session.config", ...}` | First message; sets format, voice/reference, `fastout`, etc. Same fields as the HTTP request plus `stream_audio` and `fastout`. |
| `{"type": "input.text", "text": "..."}` | Append incrementally-arriving text |
| `{"type": "input.wait"}` | Flush and speak the buffered tail now (a clause with no terminator would otherwise wait for `input.done`); keeps the session open and re-arms `fastout` for the next turn |
| `{"type": "input.stop", "chunk": K}` | Barge-in — see [Barge-in](#barge-in-inputstop) |
| `{"type": "input.done"}` | End of text for this turn; speaks any buffered tail, then the engine evicts the session |

**Server → client messages**

| Message | Purpose |
|---|---|
| `{"type": "audio.start", "sentence_index": i, "sentence_text": "...", "format": ...}` | A chunk is starting; for PCM it also carries `sample_rate` |
| binary frame(s) | Raw int16 PCM audio for the current chunk (when `stream_audio=true`) |
| `{"type": "audio.done", "sentence_index": i, "total_bytes": n, "error": false}` | The chunk finished |
| `{"type": "generation.stopped", "chunk": K}` | Acknowledges an `input.stop` |
| `{"type": "session.done", "total_sentences": n}` | The session finished (after `input.done`) |

Set `stream_audio` to `true` for progressive raw PCM binary frames. In that mode
`response_format` must be `pcm` and `speed` must be `1.0`.

**Chunking.** Each chunk is one sentence (adjacent sentences are never merged, to
preserve prosody). ASCII `.` `?` `!` split only when followed by whitespace (so
`3.14` / `U.S.A` survive); CJK `。！？` split immediately. With `fastout=true`,
the **first** chunk of a turn is released at the earliest *clause* boundary
(`,` `;` `:` as well as the sentence enders) instead of waiting for a full
sentence — this minimizes time-to-first-audio; subsequent chunks fall back to
sentence splitting. `input.wait` and `input.stop` re-arm `fastout`, so the next
turn's first chunk is again released early.

#### Barge-in (`input.stop`)

For voice agents where the user interrupts mid-utterance, send
`{"type": "input.stop", "chunk": K}`, where `K` is the `sentence_index` of the
last chunk the client actually finished **playing** (tracked from the
`audio.start` / `audio.done` events). The server:

1. aborts the in-flight chunk and drops everything still queued,
2. rolls the engine's continuity history back to `≤ K` — chunks generated ahead
   of playback but never heard are discarded,
3. rewinds the chunk index so the next synthesized chunk is `K+1`.

The contiguous index sequence lets the client discard the matching stale audio it
already buffered; both ends realign on "next chunk is `K+1`". The session stays
open, so the next `input.text` continues the same voice/session, and the server
replies `{"type": "generation.stopped", "chunk": K}`. The engine retains the full
per-session history, so `K` may be **any** previously-heard chunk — not only the
most recent few.

### Text Normalization

On every path (curl, Python, WebSocket), CJK / full-width punctuation in the
`input` text is normalized to its ASCII form just before synthesis — e.g.
`。→ .`, `，→ ,`, `！→ !`, `？→ ?`, `（）→ ()`, `“”→ "` — so the model sees one
consistent punctuation style. Spoken content is unchanged. Normalization runs
*after* sentence/chunk splitting, so it does not affect where the text is split
(boundaries are computed on the original punctuation).

## Use Python

### Voice Cloning

```python
import requests

REFERENCE_AUDIO = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
REFERENCE_TEXT = "We asked over twenty different people, and they all said it was his."
SPEECH_INPUT = "Get the trust fund to the bank early."

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": SPEECH_INPUT,
        "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": 1024,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Pre-encoded Reference Codes

For high-throughput pipelines (e.g. RL rollout) where the same reference audio is reused across many requests, you can encode the reference audio offline and pass the discrete codes directly via `reference_codes` — this skips the server-side codec encode step. Shape must be `[T, num_codebooks=8]`.

```python
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": SPEECH_INPUT,
        "reference_codes": codes_TN,  # [T, 8] int list, pre-delay-pattern
        "reference_text": REFERENCE_TEXT,
    },
)
```

### Streaming Request

```python
import base64
import json
import wave

import requests

payload = {
    "input": SPEECH_INPUT,
    "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
    "stream": True,
    "response_format": "pcm",
    "max_new_tokens": 1024,
}

pcm_chunks = []
sample_rate = 24000

with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json=payload,
    stream=True,
    timeout=600,
) as stream:
    stream.raise_for_status()
    for line in stream.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data:") :].lstrip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        audio = event.get("audio")
        if audio is None:
            continue
        sample_rate = audio.get("sample_rate", sample_rate)
        pcm_chunks.append(base64.b64decode(audio["data"]))

with wave.open("output_stream.wav", "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)  # int16 PCM
    wav.setframerate(sample_rate)
    wav.writeframes(b"".join(pcm_chunks))
```

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | string | (required) | Text to synthesize |
| `voice` | string | `"default"` | Voice identifier (ignored when `references` is set) |
| `response_format` | string | `"wav"` | Output audio format; use `"pcm"` for low-latency streaming playback |
| `stream` | bool | `false` | Enable streaming via SSE; set to `true` to receive incremental audio chunks |
| `references` | list | `null` | Reference audio for voice cloning; each item has `audio_path` (local path or HTTP URL) and `text` (transcript) |
| `ref_audio` | string | `null` | Inline reference audio: `data:` URI, raw base64, `http(s)://` / `file://` URL, or local path (alternative to `references`) |
| `ref_text` | string | `null` | Transcript of `ref_audio` |
| `reference_codes` | list[list[int]] | `null` | Pre-encoded discrete codes, shape `[T, 8]` — alternative to `references[0].audio_path` |
| `reference_text` | string | `null` | Transcript of reference audio when supplying `reference_codes` |
| `max_new_tokens` | int | `2048` | Maximum number of generated multi-codebook steps |
| `temperature` | float | `0.8` | Sampling temperature |
| `top_p` | float | `0.95` | Top-p sampling |
| `top_k` | int | `50` | Top-k sampling |
| `seed` | int | `null` | Random seed for reproducibility |

## Benchmark Results

On seed-tts en (1000 utterances) on a single A100 40GB, bf16, top_k=50, temp=0.8,
max_new_tokens=1024, scored with HF Whisper-large-v3 (fp32) for WER and
WavLM-large ECAPA-TDNN cosine similarity × 100:

| metric | value |
|---|---|
| avg WER | 0.0182 |
| avg speaker similarity | 64.81 |

Throughput (N=50/level, sequential thread pool):

| Concurrency | Mean Latency | RTF (per-req) | audio_s/s |
|---|---|---|---|
| 1 | 4637 ms | 0.526 | 1.90 |
| 16 | 7138 ms | 0.747 | 12.88 |
| 32 | 10188 ms | 0.865 | 16.94 |
