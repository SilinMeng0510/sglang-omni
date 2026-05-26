# Higgs Audio V3 TTS on sglang-omni

How to serve and call a **Higgs Audio V3** text-to-speech checkpoint through the
sglang-omni OpenAI-compatible server.

## Image

```
eigenai/sglang-omni:higgs-tts-v3
```

Built from `main` with `docker/Dockerfile.higgs-runtime` on top of
`frankleeeee/sglang-omni:dev`. All audio deps (librosa, av, silero-vad, …) are
baked in; `ENTRYPOINT` is `sgl-omni`. Pipeline stages: `preprocessing →
audio_encoder → tts_engine → vocoder`.

## Serve

```bash
docker run -d --name higgs-tts --gpus '"device=0"' \
  -v /path/to/checkpoint:/model \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  eigenai/sglang-omni:higgs-tts-v3 \
  serve --model-path /model \
        --config examples/configs/higgs_tts.yaml \
        --model-name higgs-audio-v3-tts \
        --host 0.0.0.0 --port 8000
```

A single 40 GB GPU is enough for the ~1.7B/4B TTS checkpoints. The HF cache mount
supplies the Higgs audio tokenizer (needed at load time).

Without Docker, from a checkout:

```bash
uv pip install -v -e .
sgl-omni serve --model-path /path/to/checkpoint \
  --config examples/configs/higgs_tts.yaml --port 8000
```

### Config

`examples/configs/higgs_tts.yaml` selects the Higgs TTS pipeline:

```yaml
config_cls: HiggsTtsPipelineConfig
model_path: boson-sglang/higgs-audio-v3-tts-4b-base   # overridden by --model-path
relay_backend: shm
```

## Endpoints

| Path                       | Use                                              |
| -------------------------- | ------------------------------------------------ |
| `GET  /health`             | Liveness + pipeline stage status                 |
| `GET  /v1/models`          | Served model id                                  |
| `POST /v1/audio/speech`    | TTS — returns audio (or SSE chunks when `stream`) |
| `POST /v1/chat/completions`| Chat / multimodal                                |

## Synthesize (non-streaming)

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs-audio-v3-tts",
    "input": "Hello from Higgs Audio version three.",
    "voice": "default",
    "response_format": "wav",
    "temperature": 0.7, "top_p": 0.95, "top_k": 50,
    "max_new_tokens": 2048
  }' -o out.wav
```

Output is a 24 kHz mono WAV. `response_format` also accepts `pcm/flac/mp3/aac/opus`.

## Voice cloning

Pass a reference clip (path/URL the server can read) plus its transcript:

```jsonc
{
  "model": "higgs-audio-v3-tts",
  "input": "Text to speak in the cloned voice.",
  "voice": "default",
  "ref_audio": "/data/refs/speaker.wav",
  "ref_text": "Transcript of the reference clip.",
  "response_format": "wav"
}
```

## Streaming

Set `"stream": true` to receive audio as SSE chunks as they are generated:

```bash
curl -N http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"higgs-audio-v3-tts","input":"Streaming speech.",
       "voice":"default","response_format":"pcm","stream":true}'
```

For incremental playback, `response_format: "pcm"` is required.

## Python

```python
import requests

r = requests.post("http://localhost:8000/v1/audio/speech", json={
    "model": "higgs-audio-v3-tts",
    "input": "Higgs text to speech.",
    "voice": "default",
    "response_format": "wav",
})
open("out.wav", "wb").write(r.content)
```

## UI

A Gradio TTS playground (non-streaming + streaming tabs, voice-clone controls) can
attach to a running server:

```bash
python -m playground.tts.app --api-base http://localhost:8000
```
