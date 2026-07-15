import unittest
from pathlib import Path

from common import BYTES_PER_SECOND, continuity, percentile, speech_payload, stats
from performance import load_prompts

HERE = Path(__file__).resolve().parent


class SuiteTests(unittest.TestCase):
    def test_percentile_and_stats(self):
        self.assertEqual(percentile([1, 2, 3], 50), 2)
        self.assertAlmostEqual(stats([1, 2, 3])["p85"], 2.7)
        self.assertIsNone(stats([])["mean"])

    def test_continuity_detects_underrun(self):
        # First chunk has 100 ms of audio; the next arrives after 250 ms.
        chunks = [
            (10.0, int(BYTES_PER_SECOND * 0.1)),
            (10.25, int(BYTES_PER_SECOND * 0.2)),
        ]
        jitter, speed = continuity(chunks)
        self.assertAlmostEqual(jitter, 0.15)
        self.assertAlmostEqual(speed, 0.8)

    def test_bundled_prompts_are_valid(self):
        prompts = load_prompts(HERE / "sharegpt_10k.jsonl", 1)
        self.assertEqual(len(prompts), 10_000)
        self.assertTrue(all(isinstance(value, str) and value for value in prompts))

    def test_ab_sample_texts_are_valid(self):
        samples = [
            line.strip()
            for line in (HERE / "sample_text.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(samples), 10)

    def test_performance_payload_supports_dynamic_lora(self):
        payload = speech_payload(
            "hello",
            model="higgs-tts",
            voice="default",
            seed=1,
            lora_adapter_path="/models/ap2/adapter",
        )
        self.assertEqual(payload["lora_adapter"], {"path": "/models/ap2/adapter"})


if __name__ == "__main__":
    unittest.main()
