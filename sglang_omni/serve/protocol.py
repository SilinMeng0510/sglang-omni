# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible request/response protocol definitions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Shared / Common
# ---------------------------------------------------------------------------


class UsageResponse(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Chat Completion
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str
    content: Any = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionAudio(BaseModel):
    """Audio data returned in a chat completion response."""

    id: str
    data: str  # base64-encoded audio
    expires_at: int | None = None
    transcript: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(populate_by_name=True)

    model: str | None = None
    messages: list[ChatMessage]

    # Sampling parameters
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None

    # Streaming
    stream: bool = False

    # Multi-modal output control
    modalities: list[str] | None = None  # e.g. ["text", "audio"]

    # Audio output configuration
    audio: dict[str, Any] | None = None  # {"voice": "...", "format": "wav"}

    # Audio input (sglang-omni extension)
    # Can be a list of audio file paths (local paths or URLs)
    audios: list[str] | None = None

    # Image input (sglang-omni extension)
    # Can be a list of image file paths (local paths or URLs)
    images: list[str] | None = None

    # Video input (sglang-omni extension)
    # Can be a list of video file paths (local paths or URLs)
    videos: list[str] | None = None
    video_fps: float | None = None
    video_max_frames: int | None = None
    video_min_pixels: int | None = None
    video_max_pixels: int | None = None
    video_total_pixels: int | None = None

    # Per-stage sampling overrides (sglang-omni specific)
    stage_sampling: dict[str, dict[str, Any]] | None = None
    stage_params: dict[str, dict[str, Any]] | None = None

    # Talker-specific overrides for Qwen3-Omni speech output
    talker_temperature: float | None = None
    talker_top_p: float | None = None
    talker_top_k: int | None = None
    talker_repetition_penalty: float | None = None
    talker_max_new_tokens: int | None = None

    # Misc
    request_id: str | None = None
    user: str | None = None

    @property
    def effective_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int = 0
    message: dict[str, Any]
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageResponse | None = None


class ChatCompletionStreamDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None
    audio: ChatCompletionAudio | None = None


class ChatCompletionStreamChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionStreamDelta
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]
    usage: UsageResponse | None = None


# ---------------------------------------------------------------------------
# Speech (TTS)
# ---------------------------------------------------------------------------


class SpeechReference(BaseModel):
    """Reference item for voice cloning in /v1/audio/speech."""

    audio_path: str | None = None
    text: str | None = None
    vq_codes: list[list[int]] | list[int] | None = None
    base64: str | None = None
    media_type: str | None = None


class CreateSpeechRequest(BaseModel):
    """OpenAI-compatible text-to-speech request.

    Standard OpenAI fields plus extensions for advanced TTS models
    (e.g. voice cloning, style instructions).
    """

    model_config = ConfigDict(populate_by_name=True)

    # Standard OpenAI fields
    model: str | None = None
    input: str
    voice: str = "default"
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = False

    # Advanced TTS extensions
    task_type: str | None = None  # e.g. "Base", "CustomVoice", "VoiceDesign"
    language: str | None = None
    instructions: str | None = None  # style/emotion instructions

    # Voice cloning parameters
    ref_audio: str | None = None  # path or URL to reference audio
    ref_text: str | None = None  # transcript of reference audio
    references: list[SpeechReference] | None = None  # S2-Pro-style refs
    x_vector_only_mode: bool | None = None
    speaker_embedding: list[float] | None = Field(default=None, max_length=8192)
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)

    # Generation parameters
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None

    # Per-stage overrides (sglang-omni specific)
    stage_params: dict[str, dict[str, Any]] | None = None


class StreamingSpeechSessionConfig(BaseModel):
    """Configuration sent first on /v1/audio/speech/stream."""

    model_config = ConfigDict(populate_by_name=True)

    model: str | None = None
    voice: str | None = Field(
        default=None,
        validation_alias=AliasChoices("voice", "speaker"),
    )
    task_type: str | None = None
    language: str | None = None
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)
    max_new_tokens: int | None = Field(default=None, ge=1)
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)
    ref_audio: str | None = None
    ref_text: str | None = None
    references: list[SpeechReference] | None = None
    x_vector_only_mode: bool | None = None
    speaker_embedding: list[float] | None = Field(default=None, max_length=8192)
    stream_audio: bool = False
    split_granularity: Literal["sentence", "clause"] = "sentence"
    stage_params: dict[str, dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_streaming_constraints(self) -> "StreamingSpeechSessionConfig":
        if self.stream_audio:
            if self.response_format != "pcm":
                raise ValueError(
                    "WebSocket streaming audio (stream_audio=true) requires "
                    "response_format='pcm'. Got "
                    f"response_format='{self.response_format}'."
                )
            if self.speed is None:
                self.speed = 1.0
            elif self.speed != 1.0:
                raise ValueError(
                    "Speed adjustment is not supported when stream_audio=true. "
                    "Set speed=1.0 or omit it."
                )
        return self


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------


class ModelPermission(BaseModel):
    """Model permission info."""

    id: str = "modelperm-default"
    object: str = "model_permission"
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True


class ModelCard(BaseModel):
    """A single model entry."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "sglang-omni"
    permission: list[ModelPermission] = Field(
        default_factory=lambda: [ModelPermission()]
    )
    root: str | None = None


class ModelList(BaseModel):
    """Response for GET /v1/models."""

    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)
