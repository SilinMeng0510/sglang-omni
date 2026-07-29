# Higgs TTS quality evaluation

This directory ports the SeedTTS, CV3, and MiniMax multilingual evaluation
adapters from the `higgs-mm` `jonah/higgs_multilingual_v2` branch. It evaluates
an OpenAI-compatible `/v1/audio/speech` endpoint without importing the training
repository.

The default corpus root is
`/ceph/data/higgs_audio_eval/zero_shot_tts`. The reproduction uses:

| Benchmark | Languages | Samples |
| --- | ---: | ---: |
| SeedTTS | 2 (`en`, `zh`) | 3,108 |
| CV3 | 9 | 4,500 |
| MiniMax | 23 (excludes `yue`) | 2,300 |

The normalizers follow the benchmark-specific implementations in `higgs-mm`.
Chinese is transcribed by Paraformer by default; all other languages use
Whisper large-v3. The primary result, `wer_cer_x100`, is the mean per-sample
WER/CER within each language, macro-averaged across languages, multiplied by
100. Micro WER/CER is also reported.

## Setup

Install the additional ASR and text-normalization dependencies:

```bash
pip install -e '.[tts-eval]'
```

Inspect the datasets:

```bash
python -m benchmarks.tts_eval.cli inspect seedtts
python -m benchmarks.tts_eval.cli inspect cv3
python -m benchmarks.tts_eval.cli inspect minimax
```

## Generate

Generation is incremental. Successful keys already present in a shard JSONL
are skipped, so interrupted jobs can be rerun safely.

```bash
python -m benchmarks.tts_eval.cli generate seedtts \
  --base-url http://127.0.0.1:18044 \
  --output-dir results/tts_eval/bf16/seedtts \
  --concurrency 16 \
  --seed 0
```

Reference audio is uploaded as base64 by default, which works when the endpoint
cannot see the client filesystem. Use `--reference-mode path` only when client
and server share the exact corpus path.

Multiple endpoints can share an output directory by assigning deterministic
shards:

```bash
python -m benchmarks.tts_eval.cli generate cv3 \
  --base-url http://server-a:18044 \
  --output-dir results/tts_eval/bf16/cv3 \
  --num-shards 2 --shard-index 0 --concurrency 16 --seed 0

python -m benchmarks.tts_eval.cli generate cv3 \
  --base-url http://server-b:18044 \
  --output-dir results/tts_eval/bf16/cv3 \
  --num-shards 2 --shard-index 1 --concurrency 16 --seed 0
```

## Transcribe and score

Set `CUDA_VISIBLE_DEVICES` to dedicate a GPU to each transcription process.
Use distinct transcript names if languages are split across GPUs.

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.tts_eval.cli transcribe \
  --output-dir results/tts_eval/bf16/seedtts \
  --transcript-name transcripts.jsonl \
  --device cuda:0 --batch-size 1

python -m benchmarks.tts_eval.cli score seedtts \
  --transcripts results/tts_eval/bf16/seedtts/transcripts.jsonl \
  --output results/tts_eval/bf16/seedtts/metrics.json
```

The same commands apply to `cv3` and `minimax`. To compare BF16 and FP8, keep
their generation, transcript, and metric files under separate precision
directories and use the same corpus, language list, sampling parameters, and
seed.

## Speaker similarity

Score prompt-to-generation speaker similarity with the SeedTTS
WavLM-large/ECAPA model. Multiple generation shard manifests may be supplied;
duplicate keys are removed. The result contains every per-sample score, the
sample mean, per-language means, and the language-macro mean.

```bash
PYTHONPATH=/path/to/s3prl/source:$PYTHONPATH \
python -m benchmarks.tts_eval.similarity \
  --manifest results/tts_eval/bf16/seedtts/generation_shard_000.jsonl \
  --manifest results/tts_eval/bf16/seedtts/generation_shard_001.jsonl \
  --output results/tts_eval/bf16/seedtts/similarity.json \
  --finetune-checkpoint /path/to/wavlm_large_finetune.pth \
  --wavlm-base /path/to/wavlm_large.pt \
  --device cuda
```
