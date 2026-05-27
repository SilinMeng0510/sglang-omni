# RFC: Session-level autoregressive streaming TTS (cross-chunk continuity)

**Status:** Draft / design only — no implementation yet.
**Scope:** Higgs TTS (`HiggsMultimodalQwen3`), the `/v1/audio/speech/stream`
WebSocket path. Other TTS models are out of scope for v1.

## 1. Problem

Streaming text-in over `/v1/audio/speech/stream` is consumed by a chunker
(`StreamingTextChunker`, model-owned via
`HiggsTtsPipelineConfig.create_streaming_text_splitter`) that emits one
*sentence* at a time. Today each sentence is synthesized as an **independent
request**: a fresh `request_id`, a fresh prompt, KV freed on completion. The
prompt for sentence *N* is:

```
<|tts|> [<|ref_text|> tok(ref)] <|ref_audio|> [-100]×Nref <|text|> tok(t_N) <|audio|>
```

Nothing from sentence *N-1*'s generated audio is in sentence *N*'s input. The
only thing shared across sentences is the **fixed voice reference**.

**Consequence:** voice *timbre* stays consistent (same reference), but there is
**no cross-sentence prosody continuity** — intonation, pacing, and emphasis are
decided per sentence from a cold start. For a voice agent speaking a multi-
sentence reply, this is audible (each sentence "restarts").

## 2. Goal

Generate a whole session as **one growing autoregressive sequence**, where each
chunk is conditioned on the previously generated audio of earlier chunks:

```
<|tts|> [<|ref_text|> tok(ref)] <|ref_audio|> [ref codes]
        <|text|> tok(t1) <|audio|> [a1 codes]
        <|text|> tok(t2) <|audio|> [a2 codes]
        ...
        <|text|> tok(t_k) <|audio|>          ← generate a_k here, attending to all above
```

Confirmed format (model owner):
- Prior generated audio `a_i` is fed back **exactly like the reference audio**:
  delay-pattern codes embedded via `HiggsFusedMultiTextEmbedding` at `<|audio|>`
  placeholder positions.
- **No text-level `<|eoc|>` token** is needed. The session ends when the
  WebSocket session ends (`input.done` / disconnect); the codebook-level
  `EOC_ID` still terminates each chunk's audio as today.
- Segments are **directly concatenated** (no separator token between
  `[a_i]` and the next `<|text|>`).

### Non-goals (v1)
- Non-Higgs TTS models.
- The non-streaming `POST /v1/audio/speech` path (stays one-shot; see the
  chunking-scope decision — offline is developer-managed).
- A text-level end-of-conversation token.

## 3. Why this must live engine-side (not serve)

The generated codes **never surface to the serve layer**:
- `GenerateChunk` (client-facing) carries `audio_data` (waveform), **not codes**.
- In the Higgs pipeline, `tts_engine` emits codes as an *internal* stream
  (`modality: audio_codes`) to the `vocoder` stage; the terminal payload the
  client receives is `audio_data` from the vocoder.

So "feed the previous chunk's codes back" cannot be done in the WebSocket
handler — it has only audio. The accumulation and interleaved-prompt
construction must happen **inside the Higgs pipeline**, where the codes
(`HiggsTtsState.output_codes_delayed`) and the KV cache already live. The serve
layer's only new responsibility is **session affinity** (tagging a session id).

## 4. Design

### 4.1 Session affinity (serve → engine)
- The WS handler allocates a `session_id` per connection and attaches it to
  every per-sentence request's metadata (e.g.
  `tts_params["session_id"]` / a dedicated `GenerateRequest.session_id`).
- `input.done` / disconnect signals **session end** → engine evicts session
  state (see 4.6). A `session.reset` could start a fresh sequence on the same
  connection (optional, ties into the keep-alive WS change).

### 4.2 Per-session state (in `tts_engine`)
A session store keyed by `session_id` holding the accumulated, *committed*
context:
```
SessionState:
    ref_codes_delayed: list[list[int]] | None     # the fixed voice reference (set once)
    ref_text: str | None
    segments: list[Segment]                        # appended after each chunk
Segment:
    text_token_ids: list[int]                      # tok(t_i)
    audio_codes_delayed: list[list[int]]           # a_i, delay-pattern (== output_codes_delayed)
```
- Reference is captured on the first chunk (as today).
- After a chunk finishes generating, its `output_codes_delayed` (already
  produced by the model runner) is appended as a new `Segment` alongside its
  text tokens — **inside the engine**, no round-trip to serve.

### 4.3 Interleaved prompt layout
Extend `HiggsTokenizerAdapter.build_prompt` into a builder that takes the
reference + the prior segments + the new text and emits:
- token ids:
  `[<|tts|>] [<|ref_text|> tok(ref)] [<|ref_audio|> P_ref] ( [<|text|> tok(t_i)] [<|audio|> P_i] )* [<|text|> tok(t_new)] [<|audio|>]`
  where `P_ref` / `P_i` are runs of `AUDIO_PLACEHOLDER_ID` (`-100`) sized to the
  delayed row count of the reference / segment `i`.
- an **overlay plan**: a list of `(start_offset, codes_delayed)` spans telling
  the model runner which code block to embed at which placeholder run
  (reference + every prior segment). Generation begins after the final
  `<|audio|>`.

This is the one piece that is **pure Python and unit-testable** without a GPU.

### 4.4 Model runner: multi-span overlay
Today the runner computes the fused multi-codebook embedding from
`reference_codes_delayed` and overlays it at the single `-100` run during
prefill. Generalize to **iterate the overlay plan** and overlay each span's
embedding (reference + each prior segment) at its placeholder run. Same
`HiggsFusedMultiTextEmbedding`, just N spans instead of 1.

### 4.5 Radix cache / KV isolation (correctness-critical)
Placeholder token ids (`-100`, `<|audio|>`, etc.) are **identical across
sessions**, but the *overlaid embeddings* differ per session. sglang's radix
cache keys KV on **token ids**, so naive sharing would serve one session's KV
to another → corrupt audio. Today this is avoided by namespacing per reference
via `Req.extra_key` (`build_sglang_higgs_request`).

For sessions, `extra_key` must encode the **full accumulated context** (e.g. a
hash of `ref_codes ++ all prior (text, a_i)`), so:
- a session reusing its own growing prefix **hits** cache (prefix reuse — this
  is what makes Route A efficient);
- different sessions / different histories **never** collide.

### 4.6 Lifecycle & memory
- **KV growth:** the sequence grows every chunk; KV memory and prefill cost grow
  with session length. Need a **cap** (`max_session_tokens` / max segments) and
  a policy on overflow: stop conditioning on the oldest segments (sliding
  window) or refuse/clip. Must stay within the engine `context_length` (4096
  today) — a long reply *will* hit this; sliding window over recent segments is
  the likely answer.
- **Eviction:** on `input.done`, disconnect, or idle timeout → free the session
  store entry and its radix-cache namespace.
- **Concurrency:** many sessions in flight; the store is keyed by `session_id`;
  the scheduler batches across sessions as usual.

## 5. Implementation routes

### Route A — growing prefix, re-prefill with fed-back codes (recommended)
Each chunk is still a scheduler request, but its prompt **includes all prior
segments** (text + overlaid `a_i`). The growing prefix is byte-identical to the
previous chunk's prefix plus an appended tail, so the **radix cache reuses the
shared prefix KV** (given correct `extra_key`) — effectively only the new text
tokens are prefilled, then decode. No new scheduler primitive.
- Pros: fits the request-oriented OmniScheduler; prefix reuse keeps it efficient.
- Cons: requires correct per-session `extra_key`; overlay plan rebuilt per chunk.

### Route B — persistent KV / resumable generation
Keep one request alive for the whole session; after generating `a_i`, inject the
next `<|text|> tok(t) <|audio|>` and continue decoding with KV retained.
- Pros: conceptually one true sequence; no re-prefill.
- Cons: needs a new engine capability ("append input to an in-flight generation
  and continue") that OmniScheduler/sglang does not expose today. Larger,
  riskier.

**Recommendation: Route A.**

## 6. Data flow (Route A)

```
WS: session.config → splitter built (session_id = S)
loop input.text → chunker emits sentence t_k
   → request{ text=t_k, session_id=S }  (serve; no codes here)
   → tts_engine:
        state = sessions[S]
        prompt_ids, overlay_plan = build_interleaved(state.ref, state.segments, t_k)
        runner prefill: overlay ref + a_1..a_{k-1} at their placeholder runs
        decode a_k (8 codebooks, delay pattern) until EOC
        sessions[S].segments.append(Segment(tok(t_k), a_k))   # commit
        stream a_k → vocoder → audio frames → client
input.done / disconnect → evict sessions[S]
```

## 7. Testing

**Locally unit-testable (pure Python), no GPU:**
- `build_interleaved(...)`: exact token layout for 0/1/many prior segments,
  placeholder-run sizing, overlay-plan offsets, zero-shot (no reference) case.
- Session-state accumulation: append/commit ordering, eviction.

**Requires GPU + sglang + weights (CI / model owner):**
- Multi-span overlay correctness (embeddings land at the right positions).
- Radix-cache `extra_key` isolation (no cross-session KV bleed) + prefix-reuse
  hit rate.
- Context-length / sliding-window behavior on long sessions.
- End-to-end audio quality vs the current independent-per-sentence baseline
  (the actual point: cross-chunk prosody continuity).

## 8. Rollout
- Gate behind a session/streaming mode flag; keep the current
  independent-per-sentence path as a fallback until quality is validated.
- v1: Higgs only, Route A, sliding window over recent segments within
  `context_length`.

## 9. Open questions
- Exact `extra_key` formulation (hash inputs, collision/version safety).
- Overflow policy when a session exceeds `context_length` (sliding window size?
  drop oldest text+audio segments but keep the reference?).
- Whether `session_id` belongs on `GenerateRequest` as a first-class field vs.
  inside `metadata.tts_params`.
- Interaction with the proposed keep-alive WS change (multi-turn on one
  connection) — a `session.reset` to start a new sequence without reconnecting.
