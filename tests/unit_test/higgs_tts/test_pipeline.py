# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from sglang_omni.models.higgs_tts import stages
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.utils import EOC_ID


def test_higgs_tts_engine_enables_cuda_graph_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_sglang_server_args(checkpoint_dir, context_length, **overrides):
        server_args = SimpleNamespace(
            disable_cuda_graph=overrides["disable_cuda_graph"],
            disable_overlap_schedule=False,
            max_running_requests=overrides["max_running_requests"],
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

    stages.create_sglang_tts_engine_executor("boson-sglang/higgs-audio-v3-tts-4b-base")

    assert captured["checkpoint_dir"] == "boson-sglang/higgs-audio-v3-tts-4b-base"
    assert captured["context_length"] == 4096
    assert captured["gpu_id"] == 0
    assert captured["overrides"]["disable_cuda_graph"] is False
    assert captured["overrides"]["cuda_graph_max_bs"] == 32
    assert captured["infra_disable_cuda_graph"] is True
    assert captured["server_args"].disable_cuda_graph is False
    assert captured["cuda_graph_decoder_kwargs"] == {"max_batch_size": 16}
    assert captured["init_device_graphs_called"] is True
    assert captured["server_args"].disable_overlap_schedule is True
    assert captured["adapter_kwargs"]["max_new_tokens_cap"] == 2048
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
