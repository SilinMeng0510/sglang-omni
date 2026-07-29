# SPDX-License-Identifier: Apache-2.0
"""Local Whisper/Paraformer transcription for generated benchmark audio."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def load_generation_rows(output_dir: str | Path) -> list[dict]:
    rows_by_key = {}
    for path in sorted(Path(output_dir).glob("generation_shard_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows_by_key[str(row["key"])] = row
    return list(rows_by_key.values())


def _load_completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as handle:
        completed = set()
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            transcription_success = row.get("transcription_success")
            if transcription_success is True or (
                transcription_success is None
                and row.get("is_success")
                and row.get("transcript") is not None
            ):
                completed.add(str(row["key"]))
        return completed


class WhisperTranscriber:
    def __init__(self, model_path: str, device: str) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            # Match the higgs-mm evaluator's Whisper large-v3 precision.
            torch_dtype=torch.float32,
        ).to(device)
        self.model.eval()

    def transcribe(self, paths: list[str], lang: str) -> list[str]:
        import librosa

        signals = [librosa.load(path, sr=16_000)[0] for path in paths]
        inputs = self.processor(
            signals,
            sampling_rate=16_000,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
        ).to(self.device)
        with self.torch.inference_mode():
            try:
                predicted = self.model.generate(
                    **inputs,
                    language=lang,
                    task="transcribe",
                    return_timestamps=True,
                )
            except (TypeError, ValueError):
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=lang, task="transcribe"
                )
                predicted = self.model.generate(
                    **inputs,
                    forced_decoder_ids=forced_decoder_ids,
                    return_timestamps=True,
                )
        return [
            text.strip()
            for text in self.processor.batch_decode(predicted, skip_special_tokens=True)
        ]


class ParaformerTranscriber:
    def __init__(self, model_path: str, device: str) -> None:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "Paraformer transcription needs `pip install funasr`"
            ) from exc
        device_name = "cuda" if device.startswith("cuda") else "cpu"
        self.model = AutoModel(
            model=model_path, disable_update=True, device=device_name
        )
        self.model.model.eval()

    def transcribe(self, paths: list[str], lang: str) -> list[str]:
        del lang
        try:
            import zhconv
        except ImportError as exc:
            raise RuntimeError(
                "Paraformer transcription needs `pip install zhconv`"
            ) from exc
        texts = []
        for path in paths:
            result = self.model.generate(input=path, batch_size_s=300)
            texts.append(zhconv.convert(result[0]["text"], "zh-cn").strip())
        return texts


def transcribe_generated(
    *,
    output_dir: str | Path,
    transcript_name: str,
    whisper_model: str,
    paraformer_model: str,
    device: str,
    batch_size: int,
    languages: set[str] | None = None,
    use_paraformer_for_zh: bool = True,
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_dir = Path(output_dir)
    transcript_path = output_dir / transcript_name
    completed = _load_completed(transcript_path)
    rows = [
        row
        for row in load_generation_rows(output_dir)
        if row.get("is_success")
        and row["key"] not in completed
        and (languages is None or row["lang"] in languages)
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["lang"])].append(row)

    whisper = None
    paraformer = None
    successful = failed = 0
    for lang, lang_rows in sorted(grouped.items()):
        if lang == "zh" and use_paraformer_for_zh:
            paraformer = paraformer or ParaformerTranscriber(paraformer_model, device)
            transcriber = paraformer
            effective_batch_size = 1
        else:
            whisper = whisper or WhisperTranscriber(whisper_model, device)
            transcriber = whisper
            effective_batch_size = batch_size
        logger.info("Transcribing %d %s samples", len(lang_rows), lang)
        for start in range(0, len(lang_rows), effective_batch_size):
            batch = lang_rows[start : start + effective_batch_size]
            try:
                transcripts = transcriber.transcribe(
                    [str(row["wav_path"]) for row in batch], lang
                )
                output_rows = [
                    {
                        **row,
                        "transcript": transcript,
                        "transcription_success": True,
                        "transcription_error": None,
                    }
                    for row, transcript in zip(batch, transcripts)
                ]
                successful += len(output_rows)
            except Exception as exc:
                logger.exception("Transcription failed for %s batch at %d", lang, start)
                output_rows = [
                    {
                        **row,
                        # Match higgs-mm: an ASR failure is scored as an empty
                        # hypothesis instead of silently dropping the sample.
                        "transcript": "",
                        "transcription_success": False,
                        "transcription_error": str(exc),
                    }
                    for row in batch
                ]
                failed += len(output_rows)
            with transcript_path.open("a", encoding="utf-8") as handle:
                for row in output_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
    return {
        "successful": successful,
        "failed": failed,
        "skipped_complete": len(completed),
        "transcript_path": str(transcript_path),
    }
