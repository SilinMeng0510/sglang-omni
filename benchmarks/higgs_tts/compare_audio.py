#!/usr/bin/env python3
"""Compare fixed Higgs TTS A/B samples for content and acoustic regressions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from benchmarks.higgs_tts.common import percentile, write_json


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    reference_words = normalized_words(reference)
    hypothesis_words = normalized_words(hypothesis)
    previous = list(range(len(hypothesis_words) + 1))
    for row, reference_word in enumerate(reference_words, 1):
        current = [row]
        for column, hypothesis_word in enumerate(hypothesis_words, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1], len(reference_words)


def load_audio(path: Path, target_rate: int = 16_000) -> tuple[np.ndarray, float]:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sample_rate, audio = wavfile.read(path)
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio in {path}, got {audio.shape}")
    if np.issubdtype(audio.dtype, np.integer):
        scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)
    duration = len(audio) / float(sample_rate)
    if sample_rate != target_rate:
        divisor = int(np.gcd(sample_rate, target_rate))
        audio = resample_poly(
            audio, target_rate // divisor, sample_rate // divisor
        ).astype(np.float32)
    return audio, duration


def transcribe(paths: list[Path], model_path: str, device: str) -> list[str]:
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype
    ).to(device)
    model.eval()
    transcripts = []
    for path in paths:
        audio, _ = load_audio(path)
        features = processor(
            audio, sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(device=device, dtype=dtype)
        with torch.inference_mode():
            token_ids = model.generate(
                features,
                language="en",
                task="transcribe",
                max_new_tokens=256,
            )
        transcripts.append(
            processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
        )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return transcripts


def acoustic_embeddings(
    paths: list[Path], model_path: str, device: str
) -> list[np.ndarray]:
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    extractor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype
    ).to(device)
    model.eval()
    embeddings = []
    for path in paths:
        audio, _ = load_audio(path)
        values = extractor(
            audio, sampling_rate=16_000, return_tensors="pt"
        ).input_values.to(device=device, dtype=dtype)
        with torch.inference_mode():
            embedding = model(values).last_hidden_state.float().mean(dim=1)[0]
            embedding = torch.nn.functional.normalize(embedding, dim=0)
        embeddings.append(embedding.cpu().numpy())
    return embeddings


def _micro_wer(references: list[str], hypotheses: list[str]) -> float:
    counts = [
        word_errors(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses)
    ]
    return sum(errors for errors, _ in counts) / max(
        sum(words for _, words in counts), 1
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--texts", type=Path, default=Path(__file__).with_name("sample_text.txt")
    )
    parser.add_argument("--whisper-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-paired-wer", type=float, default=0.03)
    parser.add_argument("--max-source-wer-regression", type=float, default=0.025)
    parser.add_argument("--min-embedding-mean", type=float, default=0.97)
    parser.add_argument("--min-embedding-item", type=float, default=0.94)
    parser.add_argument("--max-duration-delta-p85", type=float, default=0.25)
    args = parser.parse_args()

    texts = [
        line.strip()
        for line in args.texts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline_paths = [
        args.baseline_dir / f"{index:02d}.wav" for index in range(len(texts))
    ]
    candidate_paths = [
        args.candidate_dir / f"{index:02d}.wav" for index in range(len(texts))
    ]
    missing = [
        str(path) for path in baseline_paths + candidate_paths if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing A/B audio files: {missing}")

    baseline_transcripts = transcribe(baseline_paths, args.whisper_model, args.device)
    candidate_transcripts = transcribe(candidate_paths, args.whisper_model, args.device)
    baseline_embeddings = acoustic_embeddings(
        baseline_paths, args.embedding_model, args.device
    )
    candidate_embeddings = acoustic_embeddings(
        candidate_paths, args.embedding_model, args.device
    )

    source_wer_baseline = _micro_wer(texts, baseline_transcripts)
    source_wer_candidate = _micro_wer(texts, candidate_transcripts)
    paired_wer = _micro_wer(baseline_transcripts, candidate_transcripts)
    similarities = [
        float(np.dot(baseline, candidate))
        for baseline, candidate in zip(baseline_embeddings, candidate_embeddings)
    ]
    baseline_durations = [load_audio(path)[1] for path in baseline_paths]
    candidate_durations = [load_audio(path)[1] for path in candidate_paths]
    duration_deltas = [
        abs(candidate - baseline) / max(baseline, 1e-9)
        for baseline, candidate in zip(baseline_durations, candidate_durations)
    ]

    summary = {
        "paired_wer": paired_wer,
        "source_wer_baseline": source_wer_baseline,
        "source_wer_candidate": source_wer_candidate,
        "source_wer_regression": source_wer_candidate - source_wer_baseline,
        "embedding_cosine_mean": float(np.mean(similarities)),
        "embedding_cosine_min": float(np.min(similarities)),
        "duration_delta_p85": percentile(duration_deltas, 85),
    }
    thresholds = {
        "max_paired_wer": args.max_paired_wer,
        "max_source_wer_regression": args.max_source_wer_regression,
        "min_embedding_mean": args.min_embedding_mean,
        "min_embedding_item": args.min_embedding_item,
        "max_duration_delta_p85": args.max_duration_delta_p85,
    }
    passed = (
        paired_wer <= args.max_paired_wer
        and summary["source_wer_regression"] <= args.max_source_wer_regression
        and summary["embedding_cosine_mean"] >= args.min_embedding_mean
        and summary["embedding_cosine_min"] >= args.min_embedding_item
        and summary["duration_delta_p85"] <= args.max_duration_delta_p85
    )
    rows = [
        {
            "index": index,
            "text": text,
            "baseline_transcript": baseline_transcripts[index],
            "candidate_transcript": candidate_transcripts[index],
            "embedding_cosine": similarities[index],
            "baseline_duration_s": baseline_durations[index],
            "candidate_duration_s": candidate_durations[index],
            "duration_delta_fraction": duration_deltas[index],
        }
        for index, text in enumerate(texts)
    ]
    report = {
        "passed": passed,
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "summary": summary,
        "thresholds": thresholds,
        "rows": rows,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
