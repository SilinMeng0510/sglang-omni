# syntax=docker/dockerfile:1

# Pin the exact CUDA 13 / PyTorch 2.11 / SGLang 0.5.12.post1 image used by
# the Higgs TTS E2E runs. Override BASE_IMAGE explicitly when validating a new
# SGLang runtime rather than silently following a mutable :dev tag.
ARG BASE_IMAGE=lmsysorg/sglang-omni@sha256:46235435997d1fa93fc81fb1c2d5b7fd8470d77395a5c348c0176094ffddf95e
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="higgs-tts-sglang-omni" \
      org.opencontainers.image.description="Reproducible runtime for the Higgs TTS streaming benchmark" \
      ai.sglang-omni.e2e.base-image="lmsysorg/sglang-omni@sha256:46235435997d1fa93fc81fb1c2d5b7fd8470d77395a5c348c0176094ffddf95e"

ENV PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/workspace/sglang-omni

WORKDIR /workspace/sglang-omni
COPY . /workspace/sglang-omni

# Match the packages installed in the measured endpoint without replacing the
# base image's tuned torch, SGLang, FlashInfer, or CUDA stack.
RUN python -m pip install --no-cache-dir --break-system-packages \
      msgpack==1.2.1 \
      librosa==0.11.0 \
      qwen-vl-utils==0.0.14

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python", "-m", "sglang_omni.cli"]
CMD ["--help"]
