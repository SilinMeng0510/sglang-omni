# Higgs TTS streaming performance test

This directory contains a compact end-to-end benchmark for the OpenAI-compatible
Higgs TTS speech endpoint. It measures request throughput, generated-audio
throughput, time to first audio byte, required jitter buffer, and
continuous-playback start latency from raw PCM responses.

## Reproducible Docker environment

Build a fresh image for each code revision from the repository root:

```bash
docker build --pull \
  -f benchmarks/higgs_tts/Dockerfile \
  -t higgs-tts-sglang-omni:local .
```

The Dockerfile pins the exact measured base-image digest instead of a mutable
`dev` tag. It currently provides CUDA 13.0, PyTorch 2.11, and SGLang
0.5.12.post1. The source tree is copied into the image, while model weights,
LoRA adapters, Hugging Face cache, and benchmark outputs stay outside it.

For the 4B model, use `examples/configs/higgs_tts_4b_masked.yaml`. Start an
endpoint with one GPU and server-visible model/adapter directories:

The masked startup settings in that config are deliberately conservative and
must remain `K8/M3`: eight delayed rows before masked decode, emitting three
frames per startup chunk. Listening tests found K8/M3 audio quality acceptable,
while more aggressive masking caused audible degradation. K8/M4 was tested as
well, but its latency benefit was too small to justify further work. Treat
K8/M3 as an audio-quality invariant and optimize request handling, generation,
batching, vocoder execution, or transport instead of reducing K.

```bash
export MODEL_PATH=/absolute/path/to/higgs-tts-3-4b
export LORA_ROOT=/hot-data/checkpoints/TTSDeepclone

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

The following LoRA checkpoints are available for testing. Their request paths
are shown relative to the `/adapters` mount used above:

| Name | Host checkpoint | Request path |
| --- | --- | --- |
| `ap2` | `/hot-data/checkpoints/TTSDeepclone/c552a632c2c944d3826c8eb0d94505b6/step_02000/peft` | `/adapters/c552a632c2c944d3826c8eb0d94505b6/step_02000/peft` |
| `tpfp` | `/hot-data/checkpoints/TTSDeepclone/b640aed5d5e444f9b03642a88f348d3c/step_02000/peft` | `/adapters/b640aed5d5e444f9b03642a88f348d3c/step_02000/peft` |
| `hmbm` | `/hot-data/checkpoints/TTSDeepclone/58b6ba5df6a347558200ec8f49f0364a/step_02000/peft` | `/adapters/58b6ba5df6a347558200ec8f49f0364a/step_02000/peft` |

LoRA adapters must always be loaded dynamically: do not preload an adapter in
the server configuration. Instead, include `lora_adapter` in every API request,
using the path visible inside the container. For example, to use `ap2`:

```json
{
  "input": "Dynamic adapter test.",
  "stream": true,
  "response_format": "pcm",
  "lora_adapter": {
    "path": "/adapters/c552a632c2c944d3826c8eb0d94505b6/step_02000/peft"
  }
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
  -f benchmarks/higgs_tts/Dockerfile \
  -t higgs-tts-sglang-omni:experiment .
```

Start a server with one of the example configurations, then run:

```bash
python benchmarks/higgs_tts/performance.py \
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
LoRA loading. The benchmark includes the adapter path in every request; for
example, to test `ap2` against the container setup above:

```bash
python benchmarks/higgs_tts/performance.py \
  --base-url http://127.0.0.1:18043 \
  --model higgs-tts-4b \
  --voice default \
  --lora-adapter-path \
    /adapters/c552a632c2c944d3826c8eb0d94505b6/step_02000/peft \
  --output-dir results/higgs_tts_ap2 \
  --concurrencies 1,4,8 \
  --duration 60
```

For generated-audio regression checks, generate WAV files named `00.wav`,
`01.wav`, and so on from every line in `sample_text.txt`, using the same seed
and ap2 adapter on both revisions. The comparison gate uses Whisper content
WER, Wav2Vec2 acoustic embeddings, and duration drift:

```bash
python benchmarks/higgs_tts/compare_audio.py \
  --baseline-dir results/audio_ab/pre_rebase_ap2 \
  --candidate-dir results/audio_ab/candidate_ap2 \
  --texts benchmarks/higgs_tts/sample_text.txt \
  --whisper-model /models/whisper-small \
  --embedding-model /models/wav2vec2-base \
  --output results/audio_ab/comparison.json
```

The default gate permits at most 3% paired WER, 2.5 percentage points of WER
regression against the source text, 25% p85 duration drift, and requires
Wav2Vec2 cosine similarity of at least 0.97 on average and 0.94 per sample.

For a CPU-only plumbing check:

```bash
python benchmarks/higgs_tts/mock_server.py --port 18999
python benchmarks/higgs_tts/performance.py \
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
python -m pytest -q tests/unit_test/higgs_tts/test_benchmark_helpers.py
```
