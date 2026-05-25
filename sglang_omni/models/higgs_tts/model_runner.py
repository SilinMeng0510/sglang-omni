# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS model runner — phase-aware AR base-runner subclass.

- ``prepare_prefill``: run the model's fused multi-codebook embedding on each
  request's delayed ref codes inline, paste the result at the ``-100``
  placeholder positions, and set ``forward_batch.input_embeds``; also
  propagate ``req_ids`` so :class:`HiggsTTSModel.forward` can route per-row
  slot lookups.
- ``prepare_decode``: propagate ``req_ids`` and, when CUDA graph decode is
  enabled, copy each request's Python sampler state into fixed model buffers.
- ``post_prefill`` / ``post_decode``: read each request's newly emitted
  multi-codebook row from either slot state or graph output buffers, append
  to ``data.output_codes``, and overwrite ``result.next_token_ids`` with
  codebook-0 so the base skips its own (text-vocab) sampler.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.higgs_tts.sampler import STOP_CODE
from sglang_omni.models.higgs_tts.text_tokenizer import AUDIO_PLACEHOLDER_ID
from sglang_omni.models.higgs_tts.utils import EOC_ID

logger = logging.getLogger(__name__)


class HiggsTTSModelRunner(ModelRunner):
    """ModelRunner for :class:`HiggsTTSModel`."""

    def prepare_prefill(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        forward_batch.req_ids = [req.request_id for req in requests]
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch, requests
        )
        return None

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._collect_slot_step_outputs(result, requests)

    def prepare_decode(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        forward_batch.req_ids = [req.request_id for req in requests]
        if self._graph_decode_enabled():
            self._sync_graph_decode_state(forward_batch, requests)
        return None

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._collect_step_outputs(result, requests)

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        input_ids = forward_batch.input_ids
        device = input_ids.device
        embed_tokens = self.model.backbone.model.embed_tokens
        fused_embed = self.model.multimodal_embedding.modality_embedding_0

        # embed_tokens would OOB on -100; embed 0 first, overwrite placeholders below.
        placeholder_mask = input_ids == AUDIO_PLACEHOLDER_ID
        safe_ids = torch.where(placeholder_mask, torch.zeros_like(input_ids), input_ids)
        text_embeds = embed_tokens(safe_ids)

        offset = 0
        for sched_req in requests:
            data = sched_req.data
            end = offset + int(data.req.extend_input_len)
            codes_rows = data.reference_codes_delayed
            if not codes_rows:
                offset = end
                continue

            full_mask = placeholder_mask[offset:end]
            n_placeholders = int(full_mask.sum().item())
            if n_placeholders == 0:
                offset = end
                continue

            codes = torch.tensor(codes_rows, dtype=torch.long, device=device)
            consumed = data.num_ref_codes_consumed
            with torch.no_grad():
                embed = fused_embed(codes[consumed : consumed + n_placeholders])
            mask_idx = full_mask.nonzero(as_tuple=True)[0] + offset
            text_embeds[mask_idx] = embed.to(text_embeds.dtype)
            data.num_ref_codes_consumed = consumed + n_placeholders
            offset = end

        return text_embeds

    def _graph_decode_enabled(self) -> bool:
        return bool(getattr(self.model, "_graph_decode_ready", False))

    @staticmethod
    def _gen_params_for_row(sampling_info: Any, row: int) -> tuple[float, float, int]:
        if sampling_info is None:
            return 1.0, 1.0, 0

        def _pick(attr: str, default):
            val = getattr(sampling_info, attr, None)
            if val is None:
                return default
            item = val[row]
            return item.item() if hasattr(item, "item") else item

        temperature = float(_pick("temperatures", 1.0))
        top_p = float(_pick("top_ps", 1.0))
        top_k = int(_pick("top_ks", 0)) or 0
        return temperature, top_p, top_k

    def _sync_graph_decode_state(self, forward_batch: Any, requests: list) -> None:
        model = self.model
        batch_size = len(requests)
        max_batch_size = int(model._graph_output_codes.shape[0])
        if batch_size > max_batch_size:
            raise RuntimeError(
                f"Higgs CUDA graph decoder batch size {batch_size} exceeds "
                f"allocated graph buffer size {max_batch_size}"
            )
        if batch_size == 0:
            return

        device = model._graph_last_codes.device
        delay_counts: list[int] = []
        eoc_countdowns: list[int] = []
        generation_done: list[bool] = []
        temperatures: list[float] = []
        top_ps: list[float] = []
        top_ks: list[int] = []

        model._graph_has_last_codes[:batch_size].fill_(False)
        model._graph_last_codes[:batch_size].zero_()
        model._graph_output_codes[:batch_size].fill_(STOP_CODE)

        sampling_info = getattr(forward_batch, "sampling_info", None)
        for row_idx, sched_req in enumerate(requests):
            slot = model.get_slot(sched_req.request_id)
            sampler = slot.sampler
            last_codes = sampler.last_codes
            if last_codes is not None:
                model._graph_last_codes[row_idx].copy_(
                    last_codes.to(
                        device=device,
                        dtype=model._graph_last_codes.dtype,
                    )
                )
                model._graph_has_last_codes[row_idx] = True

            delay_counts.append(int(sampler.delay_count))
            eoc_countdowns.append(
                -1 if sampler.eoc_countdown is None else int(sampler.eoc_countdown)
            )
            generation_done.append(bool(sampler.generation_done))
            temperature, top_p, top_k = self._gen_params_for_row(sampling_info, row_idx)
            temperatures.append(temperature)
            top_ps.append(top_p)
            top_ks.append(top_k)

        model._graph_delay_count[:batch_size].copy_(
            torch.tensor(
                delay_counts, device=device, dtype=model._graph_delay_count.dtype
            )
        )
        model._graph_eoc_countdown[:batch_size].copy_(
            torch.tensor(
                eoc_countdowns, device=device, dtype=model._graph_eoc_countdown.dtype
            )
        )
        model._graph_generation_done[:batch_size].copy_(
            torch.tensor(
                generation_done,
                device=device,
                dtype=model._graph_generation_done.dtype,
            )
        )
        model._graph_temperature[:batch_size].copy_(
            torch.tensor(
                temperatures, device=device, dtype=model._graph_temperature.dtype
            )
        )
        model._graph_top_p[:batch_size].copy_(
            torch.tensor(top_ps, device=device, dtype=model._graph_top_p.dtype)
        )
        model._graph_top_k[:batch_size].copy_(
            torch.tensor(top_ks, device=device, dtype=model._graph_top_k.dtype)
        )

    def _collect_step_outputs(self, result: Any, requests: list) -> None:
        if self._graph_decode_enabled():
            self._collect_graph_step_outputs(result, requests)
        else:
            self._collect_slot_step_outputs(result, requests)

    def _collect_slot_step_outputs(self, result: Any, requests: list) -> None:
        """Pull per-request newly emitted codes from model slots into
        ``data.output_codes`` and overwrite ``result.next_token_ids`` with
        codebook-0 so the base runner skips its text-vocab sampler.
        """
        batch_size = len(requests)
        if batch_size == 0:
            return

        model = self.model
        cb0_per_row: list[int] = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            slot = model._slots.get(sched_req.request_id)
            if req.is_chunked > 0 or slot is None or not slot.output_codes:
                data.latest_stream_code_chunk = None
                cb0_per_row.append(0)
                continue
            codes_N = slot.output_codes[-1].detach().cpu().clone()
            data.output_codes.append(codes_N)
            data.generation_done = bool(slot.sampler.generation_done)
            self._mark_sampler_finished(req, data.generation_done)
            data.latest_stream_code_chunk = codes_N.unsqueeze(0)
            cb0_per_row.append(int(codes_N[0].item()))

        result.next_token_ids = torch.tensor(
            cb0_per_row,
            dtype=torch.long,
            device=result.logits_output.next_token_logits.device,
        )

    def _collect_graph_step_outputs(self, result: Any, requests: list) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return

        model = self.model
        cb0_per_row: list[int] = []
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            slot = model.get_slot(sched_req.request_id)
            self._sync_slot_from_graph_row(slot, row_idx)

            codes_N_gpu = model._graph_output_codes[row_idx].detach().clone()
            is_stop = int(codes_N_gpu[0].item()) == STOP_CODE
            if req.is_chunked > 0 or is_stop:
                data.generation_done = bool(slot.sampler.generation_done or is_stop)
                data.latest_stream_code_chunk = None
                cb0_per_row.append(0)
                continue

            slot.output_codes.append(codes_N_gpu.to(torch.long))
            codes_N = codes_N_gpu.cpu().to(torch.long)
            data.output_codes.append(codes_N)
            data.generation_done = bool(slot.sampler.generation_done)
            self._mark_sampler_finished(req, data.generation_done)
            data.latest_stream_code_chunk = codes_N.unsqueeze(0)
            cb0_per_row.append(int(codes_N[0].item()))

        result.next_token_ids = torch.tensor(
            cb0_per_row,
            dtype=torch.long,
            device=result.logits_output.next_token_logits.device,
        )

    def _sync_slot_from_graph_row(self, slot: Any, row_idx: int) -> None:
        model = self.model
        sampler = slot.sampler
        sampler.delay_count = int(model._graph_delay_count[row_idx].item())
        eoc_countdown = int(model._graph_eoc_countdown[row_idx].item())
        sampler.eoc_countdown = None if eoc_countdown < 0 else eoc_countdown
        sampler.generation_done = bool(model._graph_generation_done[row_idx].item())
        if bool(model._graph_has_last_codes[row_idx].item()):
            sampler.last_codes = (
                model._graph_last_codes[row_idx].detach().clone().to(torch.long)
            )
        else:
            sampler.last_codes = None

    @staticmethod
    def _mark_sampler_finished(req: Any, generation_done: bool) -> None:
        """Bridge Higgs sampler completion into upstream SGLang finish state."""
        if generation_done and req.finished_reason is None:
            req.finished_reason = FINISH_MATCHED_TOKEN(EOC_ID)


__all__ = ["HiggsTTSModelRunner"]
