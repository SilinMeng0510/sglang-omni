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

Final vocoder output drops the EOC-adjacent last recovered raw codec frame and
clips decoded audio to the exact remaining frame boundary. This is the default
tail behavior selected by the sample-audio listening A/B; keep it consistent
between performance and listening tests.

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

## Single-GPU dual-instance serving

At higher concurrency, one Higgs TTS process leaves scheduling gaps between
autoregressive generation, relay work, and vocoder execution. CUDA MPS can
place two complete replicas on **one physical GPU** so an external serving
layer can route requests across two independent endpoints. This is
data-parallel request serving: neither replica uses tensor parallelism or spans
multiple GPUs.

The measured H100 setup uses a total concurrency of ten, split dynamically by
the external router across two endpoints. Each endpoint captures CUDA Graphs
for batch sizes 1–5 and accepts at most five running requests. The generation
processes each receive a 50% MPS active-thread limit, while the higher-priority
vocoders receive 35%.

Start one MPS control daemon for the selected GPU. The daemon and both endpoint
processes must use the same pipe and log directories:

```bash
export CUDA_VISIBLE_DEVICES=7
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-higgs-gpu7
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-higgs-gpu7
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
```

When endpoints run in containers, bind both containers to the same physical
GPU and share the MPS pipe directory with the MPS control daemon. Do not assign
multiple physical GPUs to either endpoint.

Add the following serving-only overrides to a copy of
`examples/configs/higgs_tts_4b_masked.yaml`:

```yaml
process_env_defaults:
  pipeline:
    CUDA_MPS_CLIENT_PRIORITY: "1"
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: "50"
  vocoder:
    CUDA_MPS_CLIENT_PRIORITY: "0"
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: "35"

runtime_overrides:
  tts_engine:
    server_args_overrides:
      cuda_graph_bs: [1, 2, 3, 4, 5]
      max_running_requests: 5
```

`process_env_defaults` applies these values when each stage subprocess starts.
Priority `0` gives the vocoder preference over the generation pipeline during
token bursts. For a no-LoRA deployment, also set
`enable_dynamic_lora: false`; merely omitting `lora_adapter` from requests does
not remove the dynamic LoRA manager overhead. For request-scoped dynamic LoRA,
keep the existing `enable_dynamic_lora`, `lora_base_dir`, cache, and rank
settings.

Launch two endpoints with the same GPU and config but different ports:

```bash
python -m sglang_omni.cli serve \
  --config /path/to/higgs_tts_4b_masked_mps.yaml \
  --model-path /path/to/higgs-tts-3-4b \
  --host 0.0.0.0 \
  --port 18053 \
  --mem-fraction-static 0.32 >endpoint-0.log 2>&1 &
endpoint_0=$!

python -m sglang_omni.cli serve \
  --config /path/to/higgs_tts_4b_masked_mps.yaml \
  --model-path /path/to/higgs-tts-3-4b \
  --host 0.0.0.0 \
  --port 18063 \
  --mem-fraction-static 0.32 >endpoint-1.log 2>&1 &
endpoint_1=$!

wait "$endpoint_0" "$endpoint_1"
```

Register both endpoint URLs with the deployment's existing routing layer.
NUMA-local CPU pinning for each endpoint is optional but recommended.

No-LoRA requests need no special payload. Dynamic LoRA requests use the same
request-scoped `lora_adapter.path` described below. This setup does not merge
or quantize adapters.

### Observed H100 performance

The following results use one H100 80 GB, BF16, the 4B K8/M3 config, default
voice, the checked-in ShareGPT workload, and 120 seconds per concurrency level.
Concurrency is the total client concurrency across both replicas, not a
per-replica value. Continuous P85 is continuous-playback start latency.

No LoRA:

| Concurrency | Single audio-s/s | Dual audio-s/s | Single P85 | Dual P85 |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 28.07 | 25.76 | 101.2 ms | 92.7 ms |
| 6 | 32.35 | 30.23 | 109.6 ms | 94.2 ms |
| 7 | 36.48 | 35.15 | 119.0 ms | 97.9 ms |
| 8 | 39.46 | 40.08 | 180.7 ms | 100.2 ms |
| 9 | 40.16 | 44.44 | 326.3 ms | 100.0 ms |
| 10 | 40.24 | 48.98 | 543.8 ms | 108.0 ms |

The dual-instance topology trades some low-concurrency batching efficiency for
tail stability. It crosses the single-instance throughput curve at concurrency
eight. At concurrency ten it improves generated-audio throughput by 21.7% and
reduces Continuous P85 by 80.1%.

Request-scoped dynamic LoRA (`ap2`):

| Concurrency | Single audio-s/s | Dual audio-s/s | Single P85 | Dual P85 |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 20.09 | 17.97 | 152.9 ms | 134.6 ms |
| 6 | 23.09 | 21.35 | 154.5 ms | 135.5 ms |
| 7 | 26.42 | 24.62 | 162.2 ms | 137.4 ms |
| 8 | 29.42 | 27.79 | 162.6 ms | 147.1 ms |
| 9 | 31.91 | 30.92 | 173.5 ms | 141.9 ms |
| 10 | 34.27 | 33.90 | 184.5 ms | 157.2 ms |

Dynamic LoRA retains a small throughput advantage for one larger batch over
two smaller batches in this range. At concurrency ten the dual topology is
within 1.1% of single-instance throughput and reduces Continuous P85 by 14.8%.
Offline adapter merging is a different serving protocol and is not included in
this comparison.

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

## “Run sample audio” convention

When asked to “run sample audio”, use the 4B K8/M3 config and all ten lines in
`sample_text.txt`. Generate each line with the same per-line seed for these
three request-scoped dynamic LoRAs: `ap2`, `tpfp`, and `hmbm`. Do not preload
an adapter or change the K8/M3 startup settings. The review artifact must be an
HTML page arranged by sentence, with the three voices side by side. Each voice
card must contain the WAV player, waveform, 0–12 kHz spectrogram, duration,
and generation latency. Also save a JSON manifest containing the adapter path,
seed, relative artifact paths, and audio SHA256 for every sample.

Serve listening galleries with the range-aware helper below. Python 3.10's
standard `python -m http.server` ignores byte-range requests and prevents
reliable seeking in browser audio controls.

```bash
python benchmarks/higgs_tts/gallery_server.py \
  --port 22222 \
  --directory /path/containing/the/gallery
```

Verify seeking support by checking for `206 Partial Content`, `Accept-Ranges`,
and `Content-Range`:

```bash
curl -I -H 'Range: bytes=0-1023' \
  http://127.0.0.1:22222/gallery/ap2/00.wav
```

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
