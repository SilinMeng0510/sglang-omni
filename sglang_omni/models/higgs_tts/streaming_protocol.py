# SPDX-License-Identifier: Apache-2.0
"""Per-request streaming-TTS protocol state machine.

Mirrors the reference serving loop in higgs-mm
``higgs_mm/eval/tts/streaming.py`` (``generate_streaming_tts``), mapped onto
sglang's one-token-per-step AR decode:

- every decode step consumes ONE input (a text-space token embed or a fused
  audio-row embed) and produces ONE output (a text-head decision, an
  audio-head codebook row, or nothing);
- the sampled/forced token published as ``next_token_ids`` at step ``t`` is
  the input of step ``t+1``.

Roles (what THIS step's model output means):

- ``DECISION``: the input was an injected text token (or the opening ``T0``
  from prefill, or ``<|text_end|>``); sample the text head constrained to
  ``{<|text|>, <|audio|>}``. ``<|text|>`` = wait, ``<|audio|>`` = open an
  audio block. The text head is only consulted BEFORE the first audio block:
  training lays out every later block boundary as a forced ``<|audio|>``
  (mid-stream waits are off-distribution), so once the first block opened,
  boundaries and everything after ``<|text_end|>`` force audio.
- ``WAIT_FEED``: the input was the ``<|text|>`` wait token; output discarded.
  Publishes the next queued text token (or ``<|text_end|>``).
- ``AUDIO``: the input was ``<|audio|>`` or a previous audio row; the audio
  head samples one delayed codebook row (the existing Higgs sampler).
- ``BLOCK_END``: the input was the final row of a block; output discarded.
  Publishes the next queued text token (or ``<|text_end|>``).

The first audio block carries the delay-pattern ramp-in
(``num_extra_tokens + frames_per_block`` rows); later blocks are exactly
``frames_per_block`` rows. After ``<|text_end|>`` the model free-runs until
the audio-space EOC (handled by the existing delay/EOC sampler machinery).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class StepRole(enum.Enum):
    DECISION = "decision"
    WAIT_FEED = "wait_feed"
    AUDIO = "audio"
    BLOCK_END = "block_end"


@dataclass
class StepPlan:
    """What the NEXT decode step should do for this request.

    - ``input_token_id``: text-space token to publish as ``next_token_ids``
      (input of the next step) when ``input_is_text``; ``None`` for audio-row
      inputs (the fused embed of ``last_codes`` is used and cb0 is published).
    - ``input_is_text``: choose text embedding of the published token over the
      fused audio embedding of ``last_codes``.
    - ``audio_advance``: whether the audio sampler may advance this request's
      delay/EOC state at the next step (False freezes it for text-role and
      discard steps).
    - ``role``: the role of the next step's output.
    """

    input_token_id: int | None
    input_is_text: bool
    audio_advance: bool
    role: StepRole
    # global length fuse: no EOC within max_frames — the caller must finish
    # the request instead of scheduling another step
    hard_stop: bool = False
    # incremental input: the machine needs the next text token but the queue
    # is empty and upstream has not finished — the caller must HOLD the
    # request out of decode scheduling and call ``resume_plan()`` once
    # tokens were appended (or the queue was closed).
    starved: bool = False


_STARVED_PLAN_TEMPLATE = dict(
    input_token_id=None,
    input_is_text=True,
    audio_advance=False,
    starved=True,
)


@dataclass
class StreamingProtocolConfig:
    text_token_id: int
    audio_token_id: int
    text_end_token_id: int
    num_extra_tokens: int  # delay-pattern ramp rows = num_codebooks - 1
    frames_per_block: int = 4
    max_opening_waits: int = 64
    max_frames: int = 25 * 120  # content frames cap (excl. ramp rows)


@dataclass
class StreamingProtocolState:
    """Host-side protocol state for one streaming request.

    ``inject_text_ids`` may grow while the request runs (incremental input
    via ``append_text``); ``text_done=False`` means the upstream text is
    still arriving, so an empty queue STARVES the machine instead of
    injecting ``<|text_end|>``.
    """

    cfg: StreamingProtocolConfig
    inject_text_ids: list[int]
    text_done: bool = True  # full-text requests: queue is complete at build

    text_pos: int = 0
    role: StepRole = StepRole.DECISION  # prefill's last position outputs a decision
    text_end_sent: bool = False
    first_block_emitted: bool = False
    block_rows_remaining: int = 0
    rows_emitted: int = 0
    # True while the current block is the post-<|text_end|> free-run. A cb0
    # EOC inside a NON-tail block is off-distribution; the reference stops
    # the stream immediately there (no delay-pattern winddown), while tail
    # blocks let the sampler's EOC winddown complete normally.
    in_tail_flush: bool = False

    opening_waits: int = 0
    blocks: int = 0
    fuse_tripped: bool = False

    # protocol trace mirroring the reference implementation:
    # ('inject', id) / ('wait',) / ('audio', n) / ('text_end',) events
    trace: list = field(default_factory=list)
    _trace_audio_run: int = 0

    # ------------------------------------------------------------------
    def _flush_audio_trace(self) -> None:
        if self._trace_audio_run:
            self.trace.append(("audio", self._trace_audio_run))
            self._trace_audio_run = 0

    def append_text(self, token_ids: list[int], *, done: bool = False) -> None:
        """Incremental input: extend the inject queue / close it."""
        self.inject_text_ids.extend(int(t) for t in token_ids)
        if done:
            self.text_done = True

    @property
    def queue_empty(self) -> bool:
        return self.text_pos >= len(self.inject_text_ids)

    def _next_text_token(self) -> int | None:
        """Pop the next queued text token, <|text_end|> when the closed queue
        is exhausted, or ``None`` (starved) when an open queue is empty."""
        if self.text_pos < len(self.inject_text_ids):
            token = self.inject_text_ids[self.text_pos]
            self.text_pos += 1
            self._flush_audio_trace()
            self.trace.append(("inject", token))
            return token
        if not self.text_done:
            return None
        self.text_end_sent = True
        self._flush_audio_trace()
        self.trace.append(("text_end",))
        return self.cfg.text_end_token_id

    def _feed_plan(self) -> StepPlan:
        """Plan the injection step that follows WAIT_FEED / BLOCK_END.

        Starves (role unchanged, no state consumed) when the open queue is
        empty; ``resume_plan`` retries after tokens arrive.
        """
        token = self._next_text_token()
        if token is None:
            return StepPlan(role=self.role, **_STARVED_PLAN_TEMPLATE)
        plan = StepPlan(
            input_token_id=token,
            input_is_text=True,
            audio_advance=False,
            role=StepRole.DECISION,
        )
        self.role = plan.role
        return plan

    def resume_plan(self) -> StepPlan:
        """Re-plan after starvation once tokens were appended (or the queue
        was closed). Only valid when the previous plan had ``starved=True``."""
        return self._feed_plan()

    def _open_audio_block(self) -> StepPlan:
        """Publish <|audio|>; the next step samples the block's first row."""
        cfg = self.cfg
        self.in_tail_flush = self.text_end_sent
        if self.text_end_sent:
            # tail flush: free-run until EOC (bounded by max_frames + ramp)
            self.block_rows_remaining = max(
                cfg.max_frames + cfg.num_extra_tokens - self.rows_emitted, 1
            )
        elif not self.first_block_emitted:
            self.block_rows_remaining = cfg.num_extra_tokens + cfg.frames_per_block
        else:
            self.block_rows_remaining = cfg.frames_per_block
        self.first_block_emitted = True
        self.blocks += 1
        return StepPlan(
            input_token_id=cfg.audio_token_id,
            input_is_text=True,  # the input is the <|audio|> marker token...
            audio_advance=True,  # ...but the step's OUTPUT is the block's first row
            role=StepRole.AUDIO,
        )

    # ------------------------------------------------------------------
    def on_step_output(self, decision_is_audio: bool | None) -> StepPlan:
        """Advance the machine given THIS step's output; return the next plan.

        ``decision_is_audio`` is the constrained text-head decision for
        DECISION-role steps, ``None`` otherwise.
        """
        cfg = self.cfg
        role = self.role
        if role is StepRole.DECISION:
            if self.text_end_sent or self.first_block_emitted or decision_is_audio:
                plan = self._open_audio_block()
            else:
                # opening wait; over-budget waits trip the fuse and force audio
                self.opening_waits += 1
                if self.opening_waits > cfg.max_opening_waits:
                    self.fuse_tripped = True
                    plan = self._open_audio_block()
                else:
                    self._flush_audio_trace()
                    self.trace.append(("wait",))
                    plan = StepPlan(
                        input_token_id=cfg.text_token_id,
                        input_is_text=True,
                        audio_advance=False,
                        role=StepRole.WAIT_FEED,
                    )
        elif role is StepRole.WAIT_FEED:
            plan = self._feed_plan()
        elif role is StepRole.AUDIO:
            self.rows_emitted += 1
            self._trace_audio_run += 1
            self.block_rows_remaining -= 1
            if self.rows_emitted >= cfg.max_frames + cfg.num_extra_tokens:
                # global length fuse: no EOC in budget — finish the request
                self.fuse_tripped = True
                plan = StepPlan(
                    input_token_id=None,
                    input_is_text=False,
                    audio_advance=False,
                    role=StepRole.BLOCK_END,
                    hard_stop=True,
                )
                self.role = plan.role
                return plan
            if self.block_rows_remaining > 0:
                plan = StepPlan(
                    input_token_id=None,
                    input_is_text=False,
                    audio_advance=True,
                    role=StepRole.AUDIO,
                )
            else:
                # feed the block's last row with its output discarded
                plan = StepPlan(
                    input_token_id=None,
                    input_is_text=False,
                    audio_advance=False,
                    role=StepRole.BLOCK_END,
                )
        elif role is StepRole.BLOCK_END:
            plan = self._feed_plan()
        else:  # pragma: no cover - exhaustive enum
            raise AssertionError(f"unhandled role: {role}")

        self.role = plan.role
        return plan

    def finalize_trace(self) -> list:
        self._flush_audio_trace()
        return self.trace


__all__ = [
    "StepPlan",
    "StepRole",
    "StreamingProtocolConfig",
    "StreamingProtocolState",
]
