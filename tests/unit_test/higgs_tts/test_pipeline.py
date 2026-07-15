# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.higgs_tts import stages
from sglang_omni.models.higgs_tts.audio.utils import EOC_ID
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.proto import OmniRequest, StagePayload

_TEST_MODEL_NAME = "bosonai/higgs-tts-3-4b"


def _write_lora_metadata(adapter, model_name: str = _TEST_MODEL_NAME) -> None:
    (adapter / "export_metadata.json").write_text(
        json.dumps({"model_name": model_name}), encoding="utf-8"
    )


def test_voice_uses_public_speech_api_metadata() -> None:
    assert (
        stages._resolve_voice({"voice": "legacy"}, {"tts_params": {"voice": "chloe"}})
        == "chloe"
    )
    assert stages._resolve_voice({"voice": "legacy"}, {}) == "legacy"
    assert stages._resolve_voice({}, {"tts_params": "invalid"}) == "default"


def test_request_scoped_lora_adapter_path_uses_speech_metadata() -> None:
    metadata = {"tts_params": {"lora_adapter": {"path": " /models/ap2 "}}}

    assert stages._resolve_lora_adapter_path(metadata) == "/models/ap2"
    assert stages._resolve_lora_adapter_path({}) is None


def test_dynamic_lora_cache_loads_each_path_once(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _write_lora_metadata(adapter)
    loads = []

    def load(ref):
        loads.append(ref)
        return SimpleNamespace(success=True, error_message="")

    cache = stages.DynamicLoraCache(
        load, max_cached_adapters=2, serve_model_name=_TEST_MODEL_NAME
    )

    first_id = cache.get_or_load(str(adapter))
    second_id = cache.get_or_load(str(adapter / "."))

    assert first_id == second_id
    assert len(loads) == 1
    assert loads[0].lora_path == str(adapter.resolve())


def test_dynamic_lora_cache_rejects_new_path_when_full(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _write_lora_metadata(first)
    _write_lora_metadata(second)
    loads = []
    cache = stages.DynamicLoraCache(
        lambda ref: loads.append(ref)
        or SimpleNamespace(success=True, error_message=""),
        max_cached_adapters=1,
        serve_model_name=_TEST_MODEL_NAME,
    )

    cache.get_or_load(str(first))
    with pytest.raises(ValueError, match="cache is full"):
        cache.get_or_load(str(second))

    assert len(loads) == 1


def test_dynamic_lora_cache_surfaces_loader_failure(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _write_lora_metadata(adapter)
    cache = stages.DynamicLoraCache(
        lambda _ref: SimpleNamespace(
            success=False,
            error_message="adapter rank 64 exceeds max rank 32",
        ),
        max_cached_adapters=1,
        serve_model_name=_TEST_MODEL_NAME,
    )

    with pytest.raises(ValueError, match="rank 64 exceeds max rank 32"):
        cache.get_or_load(str(adapter))


def test_request_builder_loads_dynamic_lora_before_inference(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _write_lora_metadata(adapter)
    loads = []
    cache = stages.DynamicLoraCache(
        lambda ref: loads.append(ref)
        or SimpleNamespace(success=True, error_message=""),
        max_cached_adapters=2,
        serve_model_name=_TEST_MODEL_NAME,
    )
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(inputs="hello"),
        data=HiggsTtsState(
            prompt_token_ids=[1, 2, 3], lora_adapter_path=str(adapter)
        ).to_dict(),
    )
    wrapped = stages._with_dynamic_lora_cache(
        lambda value: HiggsTtsState.from_dict(value.data), cache
    )

    result = wrapped(payload)

    assert result.lora_id == loads[0].lora_id
    assert result.lora_adapter_path == str(adapter)


def test_higgs_tts_engine_enables_cuda_graph_by_default(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_build_sglang_server_args(checkpoint_dir, context_length, **overrides):
        server_args = SimpleNamespace(
            disable_cuda_graph=overrides["disable_cuda_graph"],
            disable_overlap_schedule=False,
            max_running_requests=overrides["max_running_requests"],
            lora_paths=overrides.get("lora_paths"),
            check_lora_server_args=lambda: captured.update(
                check_lora_server_args_called=True
            ),
        )
        captured["checkpoint_dir"] = checkpoint_dir
        captured["context_length"] = context_length
        captured["overrides"] = overrides
        captured["server_args"] = server_args
        return server_args

    def fake_create_sglang_infrastructure(server_args, gpu_id):
        captured["gpu_id"] = gpu_id
        captured["infra_disable_cuda_graph"] = server_args.disable_cuda_graph
        model = SimpleNamespace(
            reset_request=lambda _request_id: None,
            enable_cuda_graph_decoder=lambda **kwargs: captured.update(
                cuda_graph_decoder_kwargs=kwargs
            ),
        )
        return (
            SimpleNamespace(
                model_runner=SimpleNamespace(
                    model=model,
                    lora_manager=SimpleNamespace(lora_refs={}),
                    load_lora_adapter=lambda _ref: SimpleNamespace(success=True),
                    init_device_graphs=lambda: captured.update(
                        init_device_graphs_called=True
                    ),
                )
            ),
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
        )

    class FakeOutputProcessor:
        def __init__(self, **kwargs) -> None:
            captured["output_processor_kwargs"] = kwargs

    class FakeModelRunner:
        def __init__(self, model_worker, output_proc) -> None:
            captured["model_runner_args"] = (model_worker, output_proc)

    class FakeOmniScheduler:
        def __init__(self, **kwargs) -> None:
            captured["scheduler"] = "omni"
            captured["scheduler_kwargs"] = kwargs

    class FakeHiggsScheduler:
        def __init__(self, **kwargs) -> None:
            captured["scheduler"] = "higgs"
            captured["scheduler_kwargs"] = kwargs

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(
        stages, "build_sglang_server_args", fake_build_sglang_server_args
    )
    monkeypatch.setattr(
        stages, "create_sglang_infrastructure", fake_create_sglang_infrastructure
    )
    monkeypatch.setattr(stages, "truncate_rope_to_bf16", lambda model: None)
    monkeypatch.setattr(stages, "SGLangOutputProcessor", FakeOutputProcessor)
    monkeypatch.setattr(stages, "HiggsTTSModelRunner", FakeModelRunner)

    def fake_make_adapters(model, **kwargs):
        captured["adapter_kwargs"] = kwargs
        return None, None

    monkeypatch.setattr(stages, "make_higgs_scheduler_adapters", fake_make_adapters)
    monkeypatch.setattr(stages, "OmniScheduler", FakeOmniScheduler, raising=False)
    monkeypatch.setattr(stages, "HiggsScheduler", FakeHiggsScheduler, raising=False)

    # Engine-side session continuity builds a tokenizer adapter + session
    # store; stub them so the test doesn't need a real checkpoint on disk.
    monkeypatch.setattr(
        stages, "Tokenizer", SimpleNamespace(from_file=lambda _path: object())
    )
    monkeypatch.setattr(stages, "PreTrainedTokenizerFast", lambda **kwargs: object())
    monkeypatch.setattr(stages, "HiggsTokenizerAdapter", lambda _tok: object())
    monkeypatch.setattr(stages, "SessionStore", lambda **kwargs: object())

    adapter = tmp_path / "laura-a"
    adapter.mkdir()
    _write_lora_metadata(adapter)
    stages.create_sglang_tts_engine_executor(
        "boson-sglang/higgs-audio-v3-tts-4b-base",
        lora_voices={"Laura A": str(adapter)},
        serve_model_name=_TEST_MODEL_NAME,
        enable_dynamic_lora=True,
        lora_max_cached_adapters=3,
    )

    assert captured["checkpoint_dir"] == "boson-sglang/higgs-audio-v3-tts-4b-base"
    assert captured["context_length"] == 4096
    assert captured["gpu_id"] == 0
    assert captured["overrides"]["disable_cuda_graph"] is False
    assert captured["overrides"]["cuda_graph_max_bs"] == 32
    assert captured["overrides"]["max_running_requests"] == 16
    assert captured["overrides"]["enable_lora"] is True
    assert captured["overrides"]["lora_paths"] == {"Laura A": str(adapter)}
    assert captured["overrides"]["max_lora_rank"] == 32
    assert captured["overrides"]["max_loaded_loras"] == 4
    assert captured["overrides"]["max_loras_per_batch"] == 4
    assert captured["overrides"]["lora_backend"] == "triton"
    assert captured["overrides"]["lora_target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert captured["check_lora_server_args_called"] is True
    assert captured["infra_disable_cuda_graph"] is True
    assert captured["server_args"].disable_cuda_graph is False
    assert captured["cuda_graph_decoder_kwargs"] == {"max_batch_size": 16}
    assert captured["init_device_graphs_called"] is True
    assert captured["server_args"].disable_overlap_schedule is True
    assert captured["adapter_kwargs"]["max_new_tokens_cap"] == 1024
    # Engine-side continuity wiring: a tokenizer adapter + session store are
    # handed to the scheduler adapters.
    assert captured["adapter_kwargs"]["adapter"] is not None
    assert captured["adapter_kwargs"]["session_store"] is not None
    assert captured["scheduler"] == "omni"
    assert captured["scheduler_kwargs"]["tp_worker"] is captured["model_runner_args"][0]
    assert callable(captured["scheduler_kwargs"]["stream_output_builder"])


def test_higgs_model_runner_marks_sampler_finish() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    slot = SimpleNamespace(
        output_codes=[torch.tensor([EOC_ID, 1, 2])],
        sampler=SimpleNamespace(generation_done=True),
    )
    runner.model = SimpleNamespace(
        _slots={"req": slot},
        _graph_decode_ready=False,
    )
    req = SimpleNamespace(is_chunked=0, finished_reason=None)
    data = SimpleNamespace(req=req, output_codes=[], generation_done=False)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(1, 4))
    )

    runner._collect_step_outputs(
        result,
        [SimpleNamespace(request_id="req", data=data)],
    )

    assert data.generation_done is True
    assert req.finished_reason is not None
    assert len(data.output_codes) == 1
    assert torch.equal(data.latest_stream_code_chunk, torch.tensor([[EOC_ID, 1, 2]]))
    assert result.next_token_ids.tolist() == [EOC_ID]


def test_higgs_model_runner_marks_sampler_finish_graph() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    slot = SimpleNamespace(
        output_codes=[],
        sampler=SimpleNamespace(
            delay_count=0,
            eoc_countdown=None,
            generation_done=False,
            last_codes=None,
        ),
    )
    runner.model = SimpleNamespace(
        _graph_decode_ready=True,
        _graph_output_codes=torch.tensor([[EOC_ID, 1, 2]]),
        _graph_delay_count=torch.tensor([8], dtype=torch.int32),
        _graph_eoc_countdown=torch.tensor([0], dtype=torch.int32),
        _graph_generation_done=torch.tensor([True]),
        _graph_has_last_codes=torch.tensor([True]),
        _graph_last_codes=torch.tensor([[1, 2, 3]]),
        _graph_sampling_seed=torch.tensor([1234], dtype=torch.long),
        _graph_sampling_step=torch.tensor([9], dtype=torch.long),
        get_slot=lambda _rid: slot,
    )
    req = SimpleNamespace(is_chunked=0, finished_reason=None)
    data = SimpleNamespace(req=req, output_codes=[], generation_done=False)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(1, 4))
    )

    runner._collect_step_outputs(
        result,
        [SimpleNamespace(request_id="req", data=data)],
    )

    assert data.generation_done is True
    assert req.finished_reason is not None
    assert len(data.output_codes) == 1
    assert torch.equal(data.latest_stream_code_chunk, torch.tensor([[EOC_ID, 1, 2]]))
    assert result.next_token_ids.tolist() == [EOC_ID]


def test_higgs_performance_config_routes_lora_and_vocoder_batching() -> None:
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    cfg = HiggsTtsPipelineConfig(
        model_path="test-model",
        lora_voices={"Laura A": "/models/laura-a"},
        lora_backend="csgmv",
        lora_max_rank=64,
        enable_dynamic_lora=True,
        lora_max_cached_adapters=6,
        separate_vocoder_process=True,
        vocoder_max_batch_size=16,
        vocoder_audio_chunk_size=32,
        vocoder_audio_chunk_overlap_size=16,
    )
    stages_by_name = {stage.name: stage for stage in cfg.stages}

    assert stages_by_name["preprocessing"].factory_args["lora_voices"] == {
        "Laura A": "/models/laura-a"
    }
    assert stages_by_name["tts_engine"].factory_args["lora_voices"] == {
        "Laura A": "/models/laura-a"
    }
    assert stages_by_name["tts_engine"].factory_args["lora_backend"] == "csgmv"
    assert stages_by_name["tts_engine"].factory_args["lora_max_rank"] == 64
    assert stages_by_name["tts_engine"].factory_args["enable_dynamic_lora"] is True
    assert stages_by_name["tts_engine"].factory_args["lora_max_cached_adapters"] == 6
    assert stages_by_name["tts_engine"].factory_args["serve_model_name"] == "test-model"
    assert stages_by_name["vocoder"].factory_args["max_batch_size"] == 16
    assert stages_by_name["vocoder"].factory_args["audio_chunk_size"] == 32
    assert stages_by_name["vocoder"].factory_args["audio_chunk_overlap_size"] == 16
    assert stages_by_name["vocoder"].process == "vocoder"
    assert (
        sum(
            stages_by_name[name].runtime.resources.total_gpu_memory_fraction
            for name in ("audio_encoder", "tts_engine", "vocoder")
        )
        == 1.0
    )


def test_separate_vocoder_preserves_explicit_memory_fractions() -> None:
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    stages = HiggsTtsPipelineConfig(model_path="test-model").stages
    expected = {"audio_encoder": 0.02, "tts_engine": 0.91, "vocoder": 0.07}
    for stage in stages:
        if stage.name in expected:
            stage.runtime.resources.total_gpu_memory_fraction = expected[stage.name]

    cfg = HiggsTtsPipelineConfig(
        model_path="test-model",
        stages=stages,
        separate_vocoder_process=True,
    )

    actual = {
        stage.name: stage.runtime.resources.total_gpu_memory_fraction
        for stage in cfg.stages
        if stage.name in expected
    }
    assert actual == expected


def test_audio_encoder_defers_codec_load_until_raw_reference(monkeypatch) -> None:
    load_calls = []
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages,
        "Tokenizer",
        SimpleNamespace(from_file=lambda _path: object()),
    )
    monkeypatch.setattr(
        stages,
        "PreTrainedTokenizerFast",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(stages, "HiggsTokenizerAdapter", lambda _tokenizer: object())
    monkeypatch.setattr(
        stages,
        "get_or_load_codec",
        lambda *args: load_calls.append(args),
    )

    stages.create_audio_encoder_executor("test-model")

    assert load_calls == []
