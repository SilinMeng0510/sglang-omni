# Higgs TTS streaming performance test

This directory contains a compact end-to-end benchmark for the OpenAI-compatible
Higgs TTS speech endpoint. It measures request throughput, generated-audio
throughput, time to first audio byte, required jitter buffer, and
continuous-playback start latency from raw PCM responses.

## Reproducible Docker environment

Build a fresh image for each code revision from the repository root:

```bash
docker build --pull \
  -f tests/e2e/higgs_tts/Dockerfile \
  -t higgs-tts-sglang-omni:local .
```

The Dockerfile pins the exact measured base-image digest instead of a mutable
`dev` tag. It currently provides CUDA 13.0, PyTorch 2.11, and SGLang
0.5.12.post1. The source tree is copied into the image, while model weights,
LoRA adapters, Hugging Face cache, and benchmark outputs stay outside it.

Start an endpoint with one GPU and server-visible model/adapter directories:

```bash
export MODEL_PATH=/absolute/path/to/higgs-tts-3-4b
export LORA_ROOT=/absolute/path/to/lora-adapters

docker run --rm --name higgs-tts-sglang-omni \
  --gpus '"device=0"' \
  --shm-size 16g \
  -p 18043:18043 \
  -v "$MODEL_PATH:/models/higgs-tts-3-4b:ro" \
  -v "$LORA_ROOT:/adapters:ro" \
  higgs-tts-sglang-omni:local \
  serve \
  --config examples/configs/higgs_tts_4b_masked.yaml \
  --model-path /models/higgs-tts-3-4b \
  --host 0.0.0.0 \
  --port 18043
```

Use paths as seen inside the container in API requests, for example:

```json
{
  "input": "Dynamic adapter test.",
  "stream": true,
  "response_format": "pcm",
  "lora_adapter": {"path": "/adapters/ap2_4b"}
}
```

To identify a running environment later:

```bash
docker inspect higgs-tts-sglang-omni --format '{{.Config.Image}}'
docker image inspect higgs-tts-sglang-omni:local \
  --format '{{json .Config.Labels}} {{json .RepoDigests}}'
```

Changing SGLang/CUDA is an explicit experiment. Override the pinned base only
at build time and record the resulting digest with the benchmark output:

```bash
docker build --pull \
  --build-arg BASE_IMAGE=registry/image@sha256:... \
  -f tests/e2e/higgs_tts/Dockerfile \
  -t higgs-tts-sglang-omni:experiment .
```

Start a server with one of the example configurations, then run:

```bash
python tests/e2e/higgs_tts/performance.py \
  --base-url http://127.0.0.1:18043 \
  --model higgs-tts-4b \
  --voice default \
  --output-dir results/higgs_tts \
  --concurrencies 1,4,8 \
  --duration 60
```

The benchmark defaults to the checked-in `sharegpt_10k.jsonl` workload so runs
are reproducible. Use `--prompts /path/to/workload.jsonl` to override it; each
JSONL row must contain a non-empty `text` or `prompt` field. `sample_text.txt`
contains the fixed utterances used for generated-audio A/B listening tests.
Pass `--lora-adapter-path /path/to/adapter` to benchmark request-scoped dynamic
LoRA loading. The first request loads the adapter and later requests exercise
the cache-hit path.

For a CPU-only plumbing check:

```bash
python tests/e2e/higgs_tts/mock_server.py --port 18999
python tests/e2e/higgs_tts/performance.py \
  --base-url http://127.0.0.1:18999 \
  --model mock \
  --output-dir results/higgs_tts_mock \
  --concurrencies 1,4 \
  --duration 2
```

The production endpoint returns headerless mono signed 16-bit little-endian PCM
at 24 kHz for `stream=true` and `response_format=pcm`. Results are written under
the requested output directory and are ignored by Git.

Run the local helper tests with:

```bash
python tests/e2e/higgs_tts/test_suite.py
```
