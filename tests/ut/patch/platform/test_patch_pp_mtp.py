# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.model import ModelConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

from vllm_ascend.patch.platform.patch_pp_mtp import (
    _apply_patch,
    _is_pd_prefill_node,
    _patch_engine_core,
    _patch_model_config_validation,
    _patch_model_runner_output,
    _patch_scheduler_make_cached_request_data,
    _patch_scheduler_update_after_schedule,
    _patch_scheduler_update_from_output,
    _update_pp_mtp_spec_token_ids,
    _use_pp_decode_fence,
    _use_pp_ipc_runtime_patch,
    _use_pp_mtp_runtime_patch,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.mark.parametrize(
    "model_type,architecture",
    [
        ("qwen3_5_mtp", "Qwen3_5MTP"),
        ("glm5_next_mtp", "Glm5NextMTPModel"),
        ("qwen3", "DFlashDraftModel"),
        ("qwen3", "DFlash2DraftModel"),
        ("qwen3", "DSparkDraftModel"),
        ("qwen3", "Qwen3DSparkModel"),
        ("kimi_k3", "K3DSparkModel"),
    ],
)
def test_model_config_validates_local_drafter_as_single_pp_rank(monkeypatch, model_type, architecture):
    fake_registry = SimpleNamespace(
        is_pp_supported_model=lambda _architectures, _model_config: False,
    )
    monkeypatch.setattr(ModelConfig, "registry", property(lambda _self: fake_registry))

    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(model_type=model_type)
    model_config.runner = "draft"
    model_config.model_arch_config = SimpleNamespace(
        total_num_attention_heads=1,
        architectures=[architecture],
    )
    model_config.multimodal_config = None

    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        pipeline_parallel_size=2,
        decode_context_parallel_size=1,
    )

    ModelConfig.verify_with_parallel_config(model_config, parallel_config)
    assert parallel_config.pipeline_parallel_size == 2


def test_model_config_keeps_target_model_pp_validation(monkeypatch):
    fake_registry = SimpleNamespace(
        is_pp_supported_model=lambda _architectures, _model_config: False,
    )
    monkeypatch.setattr(ModelConfig, "registry", property(lambda _self: fake_registry))

    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(model_type="qwen3_5_mtp")
    model_config.runner = "generate"
    model_config.model_arch_config = SimpleNamespace(
        total_num_attention_heads=1,
        architectures=["UnsupportedForPP"],
    )

    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        pipeline_parallel_size=2,
        decode_context_parallel_size=1,
    )

    with pytest.raises(NotImplementedError):
        ModelConfig.verify_with_parallel_config(model_config, parallel_config)


@pytest.mark.parametrize(
    (
        "use_pp",
        "speculative_config",
        "async_scheduling",
        "use_v2_model_runner",
        "expected",
    ),
    [
        (True, object(), False, False, True),
        (True, None, True, False, True),
        (True, None, False, False, True),
        (False, object(), True, False, False),
        (True, object(), True, True, False),
    ],
)
def test_pp_ipc_runtime_patch_enabled_for_all_v1_pp(
    use_pp,
    speculative_config,
    async_scheduling,
    use_v2_model_runner,
    expected,
):
    vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=speculative_config,
        use_v2_model_runner=use_v2_model_runner,
    )

    assert _use_pp_ipc_runtime_patch(vllm_config, use_pp) is expected


def test_pp_ipc_runtime_patch_skips_pd_prefill_node():
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=True,
            is_kv_consumer=False,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=object(),
        use_v2_model_runner=False,
    )

    assert _use_pp_ipc_runtime_patch(vllm_config, use_pp=True) is False


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_pp_ipc_cached_request_data_carries_confirmed_token_for_sync_and_async(
    async_scheduling,
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.use_pp = True
    scheduler.use_v2_model_runner = False
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
    scheduler.vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        speculative_config=object(),
        use_v2_model_runner=False,
    )
    scheduler.prev_step_scheduled_req_ids = set()

    request = SimpleNamespace(
        request_id="req-0",
        all_token_ids=[11, 12, 13],
        num_computed_tokens=2,
        num_output_tokens=1,
        num_output_placeholders=0,
    )
    blocks = SimpleNamespace(get_block_ids=lambda allow_none: ([0],))

    cached_reqs_data = Scheduler._make_cached_request_data(
        scheduler,
        running_reqs=[request],
        resumed_reqs=[],
        num_scheduled_tokens={"req-0": 3},
        spec_decode_tokens={"req-0": [101, 102]},
        req_to_new_blocks={"req-0": blocks},
    )

    assert cached_reqs_data.req_ids == ["req-0"]
    assert cached_reqs_data.new_token_ids == [[13]]
    assert scheduler.scheduler_config.async_scheduling is async_scheduling


@pytest.mark.parametrize(
    ("async_scheduling", "expected_new_token_ids"),
    [(False, [[]]), (True, [[13]])],
)
def test_pp_ipc_cached_request_data_fills_empty_confirmed_token_only_for_async(
    async_scheduling,
    expected_new_token_ids,
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.use_pp = True
    scheduler.use_v2_model_runner = False
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
    scheduler.vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        speculative_config=object(),
        use_v2_model_runner=False,
    )
    scheduler.prev_step_scheduled_req_ids = set()

    request = SimpleNamespace(
        request_id="req-0",
        all_token_ids=[11, 12, 13],
        num_computed_tokens=3,
        num_output_tokens=1,
        num_output_placeholders=0,
    )
    blocks = SimpleNamespace(get_block_ids=lambda allow_none: ([0],))

    cached_reqs_data = Scheduler._make_cached_request_data(
        scheduler,
        running_reqs=[request],
        resumed_reqs=[],
        num_scheduled_tokens={"req-0": 2},
        spec_decode_tokens={"req-0": [101, 102]},
        req_to_new_blocks={"req-0": blocks},
    )

    assert cached_reqs_data.new_token_ids == expected_new_token_ids
    assert scheduler.scheduler_config.async_scheduling is async_scheduling


def test_pp_ipc_sampled_token_handoff_advances_async_non_last_rank_state(
    monkeypatch,
):
    monkeypatch.setattr(
        "vllm_ascend.worker.model_runner_v1.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False),
    )

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.is_kv_producer = False
    runner.is_kv_consumer = False
    runner.use_async_scheduling = True
    runner.device = torch.device("cpu")
    runner.discard_request_mask = SimpleNamespace(
        np=np.zeros(2, dtype=bool),
    )
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["req-0", "req-1"],
        prev_sampled_token_ids=None,
        prev_req_id_to_index={},
        num_tokens_no_spec=np.array([3, 5], dtype=np.int64),
        is_token_ids=np.zeros((2, 8), dtype=bool),
    )
    runner.requests = {
        "req-0": SimpleNamespace(output_token_ids=[31]),
        "req-1": SimpleNamespace(output_token_ids=[41, 42]),
    }
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-0", "req-1"],
            new_token_ids=[[101], [202]],
            num_output_tokens=[1, 2],
        ),
    )

    runner._apply_pp_sampled_tokens_from_scheduler_output(scheduler_output)

    assert runner.input_batch.prev_req_id_to_index == {
        "req-0": 0,
        "req-1": 1,
    }
    assert runner.input_batch.prev_sampled_token_ids.tolist() == [[101], [202]]
    assert runner.requests["req-0"].output_token_ids == [
        31,
        PLACEHOLDER_TOKEN_ID,
    ]
    assert runner.requests["req-1"].output_token_ids == [
        41,
        42,
        PLACEHOLDER_TOKEN_ID,
    ]
    assert runner.input_batch.is_token_ids[0, 3]
    assert runner.input_batch.is_token_ids[1, 5]
    assert runner.input_batch.num_tokens_no_spec.tolist() == [4, 6]


def test_pp_ipc_sampled_token_handoff_keeps_sync_path_on_scheduler_tokens(
    monkeypatch,
):
    monkeypatch.setattr(
        "vllm_ascend.worker.model_runner_v1.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False),
    )

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.is_kv_producer = False
    runner.is_kv_consumer = False
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["req-0"],
        prev_sampled_token_ids="keep",
        prev_req_id_to_index={"keep": 0},
        num_tokens_no_spec=np.array([3], dtype=np.int64),
        is_token_ids=np.zeros((1, 8), dtype=bool),
    )
    runner.requests = {
        "req-0": SimpleNamespace(output_token_ids=[31]),
    }
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-0"],
            new_token_ids=[[101]],
            num_output_tokens=[1],
        ),
    )

    runner._apply_pp_sampled_tokens_from_scheduler_output(scheduler_output)

    assert runner.input_batch.prev_req_id_to_index == {"keep": 0}
    assert runner.input_batch.prev_sampled_token_ids == "keep"
    assert runner.requests["req-0"].output_token_ids == [31]
    assert not runner.input_batch.is_token_ids[0, 3]
    assert runner.input_batch.num_tokens_no_spec.tolist() == [3]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_pp_mtp_spec_tokens_are_written_from_model_runner_output_for_sync_and_async(
    async_scheduling,
):
    request = SimpleNamespace(
        spec_token_ids=[],
        structured_output_request=None,
        is_finished=lambda: False,
    )
    scheduler = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        requests={"req-0": request},
        structured_output_manager=SimpleNamespace(
            should_advance=lambda _request: False,
        ),
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-0": 1})
    model_runner_output = SimpleNamespace(
        req_id_to_index={"req-0": 0},
        sampled_token_ids=[[200]],
        spec_token_ids=[[301, 302]],
    )

    _update_pp_mtp_spec_token_ids(
        scheduler,
        scheduler_output,
        model_runner_output,
    )

    assert request.spec_token_ids == [301, 302]


def test_pp_mtp_helpers_and_patch_reentry():
    assert _is_pd_prefill_node(SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_role="kv_producer"))) is True
    vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        speculative_config=object(),
        use_v2_model_runner=False,
    )
    assert _use_pp_mtp_runtime_patch(vllm_config, use_pp=True) is True
    assert _use_pp_mtp_runtime_patch(vllm_config, use_pp=False) is False
    vllm_config.speculative_config = None
    assert _use_pp_mtp_runtime_patch(vllm_config, use_pp=True) is False

    _apply_patch()
    _patch_model_runner_output()
    _patch_engine_core()
    _patch_scheduler_update_after_schedule()
    _patch_scheduler_make_cached_request_data()
    _patch_scheduler_update_from_output()
    _patch_model_config_validation()


def test_pp_mtp_post_step_skips_inflight_batch():
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(
            vllm_config=SimpleNamespace(
                kv_transfer_config=None,
                speculative_config=object(),
                use_v2_model_runner=False,
            ),
            use_pp=True,
        ),
        batch_queue=object(),
        async_scheduling=False,
        use_spec_decode=True,
    )
    assert EngineCore.post_step(engine, True) is None


def test_pp_mtp_spec_token_id_edge_cases():
    finished = SimpleNamespace(is_finished=lambda: True, spec_token_ids=["keep"])
    grammar = SimpleNamespace(validate_tokens=lambda tokens: tokens[:1])
    advancing = SimpleNamespace(
        spec_token_ids=[],
        is_finished=lambda: False,
        structured_output_request=SimpleNamespace(grammar=grammar),
    )
    empty_sampled = SimpleNamespace(
        spec_token_ids=["keep"],
        is_finished=lambda: False,
    )
    scheduler = SimpleNamespace(
        requests={"fin": finished, "adv": advancing, "empty": empty_sampled},
        structured_output_manager=SimpleNamespace(should_advance=lambda req: req is advancing),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"missing": 1, "fin": 1, "no-index": 1, "empty": 1, "adv": 1},
    )
    model_runner_output = SimpleNamespace(
        spec_token_ids=[[301, 302], [401, 402], [501, 502]],
        sampled_token_ids=[[], [200], [200]],
        req_id_to_index={"fin": 0, "empty": 0, "adv": 1},
    )
    _update_pp_mtp_spec_token_ids(scheduler, scheduler_output, SimpleNamespace(spec_token_ids=None))
    _update_pp_mtp_spec_token_ids(scheduler, scheduler_output, model_runner_output)
    assert finished.spec_token_ids == ["keep"]
    assert empty_sampled.spec_token_ids == []
    assert advancing.spec_token_ids == [401]


def test_pp_ipc_cached_request_data_skips_empty_output_tokens():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.use_pp = True
    scheduler.use_v2_model_runner = False
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=True)
    scheduler.vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        speculative_config=object(),
        use_v2_model_runner=False,
    )
    scheduler.prev_step_scheduled_req_ids = set()
    req_empty = SimpleNamespace(
        request_id="req-0",
        all_token_ids=[],
        num_computed_tokens=3,
        num_output_tokens=0,
        num_output_placeholders=0,
    )
    blocks = SimpleNamespace(get_block_ids=lambda allow_none: ([0],))
    cached = Scheduler._make_cached_request_data(
        scheduler,
        running_reqs=[req_empty],
        resumed_reqs=[],
        num_scheduled_tokens={"req-0": 2},
        spec_decode_tokens={},
        req_to_new_blocks={"req-0": blocks},
    )
    assert cached.new_token_ids == [[]]


@pytest.mark.parametrize(
    "use_pp,use_v2,speculative,async_scheduling,expected_fence",
    [
        (True, True, True, False, True),
        (True, True, True, True, False),
        (True, True, False, False, False),
        (False, True, True, False, False),
        (True, False, True, False, True),
    ],
)
def test_pp_decode_waits_for_sampled_output(monkeypatch, use_pp, use_v2, speculative, async_scheduling, expected_fence):
    # The final prefill produces a draft before the PP queue consumes its
    # sampled output. Only a synchronous speculative request needs fencing.
    request = SimpleNamespace(is_prefill_chunk=True, next_decode_eligible_step=0, spec_token_ids=[301, 302, 303])
    chunked_request = SimpleNamespace(is_prefill_chunk=True, next_decode_eligible_step=0)
    config = SimpleNamespace(
        kv_transfer_config=None,
        use_v2_model_runner=use_v2,
        speculative_config=SimpleNamespace(method="mtp") if speculative else None,
        model_config=SimpleNamespace(architecture="Glm5NextForConditionalGeneration"),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )
    scheduler = SimpleNamespace(
        use_pp=use_pp,
        vllm_config=config,
        requests={"final": request, "chunked": chunked_request},
    )
    output = SimpleNamespace(num_scheduled_tokens={"final": 19, "chunked": 8})
    sentinel = object()

    def update_after_schedule(_scheduler, _output):
        # Mirrors the upstream transition at the end of the prompt.
        request.is_prefill_chunk = False

    monkeypatch.setattr(Scheduler, "_update_after_schedule", update_after_schedule)
    monkeypatch.setattr(Scheduler, "update_from_output", lambda *_args: sentinel)
    _patch_scheduler_update_after_schedule()
    _patch_scheduler_update_from_output()

    Scheduler._update_after_schedule(scheduler, output)
    assert (request.next_decode_eligible_step > 1) is expected_fence
    assert chunked_request.next_decode_eligible_step == 0

    # Native MRV2 draft ids stay intact; sampled output releases the request.
    result = Scheduler.update_from_output(scheduler, output, SimpleNamespace(spec_token_ids=None))
    assert result is sentinel
    assert request.next_decode_eligible_step == 0
    assert request.spec_token_ids == [301, 302, 303]


def test_mrv2_earlier_prefill_output_keeps_later_decode_fence(monkeypatch):
    request = SimpleNamespace(is_prefill_chunk=True, next_decode_eligible_step=0)
    scheduler = SimpleNamespace(
        use_pp=True,
        vllm_config=SimpleNamespace(
            kv_transfer_config=None,
            use_v2_model_runner=True,
            speculative_config=SimpleNamespace(method="mtp"),
            model_config=SimpleNamespace(architecture="Glm5NextForConditionalGeneration"),
            parallel_config=SimpleNamespace(pipeline_parallel_size=2),
            scheduler_config=SimpleNamespace(async_scheduling=False),
        ),
        requests={"req": request},
    )
    monkeypatch.setattr(Scheduler, "_update_after_schedule", lambda *_args: None)
    monkeypatch.setattr(Scheduler, "update_from_output", lambda *_args: None)
    _patch_scheduler_update_after_schedule()
    _patch_scheduler_update_from_output()

    earlier_chunk = SimpleNamespace(num_scheduled_tokens={"req": 4096})
    Scheduler._update_after_schedule(scheduler, earlier_chunk)
    assert request.next_decode_eligible_step == 0

    request.is_prefill_chunk = False
    final_chunk = SimpleNamespace(num_scheduled_tokens={"req": 1733})
    Scheduler._update_after_schedule(scheduler, final_chunk)
    assert request.next_decode_eligible_step > 1

    sampled_output = SimpleNamespace(spec_token_ids=None)
    Scheduler.update_from_output(scheduler, earlier_chunk, sampled_output)
    assert request.next_decode_eligible_step > 1
    Scheduler.update_from_output(scheduler, final_chunk, sampled_output)
    assert request.next_decode_eligible_step == 0


@pytest.mark.parametrize("method", ["mtp", "dflash", "eagle3", "dspark"])
@pytest.mark.parametrize(
    "architecture", ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration", "Qwen3_5ForCausalLM"]
)
def test_mrv2_decode_fence_only_uses_registered_capabilities(method, architecture):
    config = SimpleNamespace(
        use_v2_model_runner=True,
        kv_transfer_config=None,
        model_config=SimpleNamespace(architecture=architecture),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        speculative_config=SimpleNamespace(method=method),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    assert _use_pp_decode_fence(config, True) == (method in ("mtp", "dflash") and architecture.startswith("Glm5Next"))
    config.kv_transfer_config = SimpleNamespace(kv_role="kv_producer")
    assert not _use_pp_decode_fence(config, True)


@pytest.mark.parametrize("transition", ["cancel", "reuse_id", "reschedule"])
def test_mrv2_stale_output_cannot_release_another_decode_fence(monkeypatch, transition):
    request = SimpleNamespace(is_prefill_chunk=False, next_decode_eligible_step=0)
    config = SimpleNamespace(
        use_v2_model_runner=True,
        kv_transfer_config=None,
        model_config=SimpleNamespace(architecture="Glm5NextForCausalLM"),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        speculative_config=SimpleNamespace(method="dflash"),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    scheduler = SimpleNamespace(use_pp=True, vllm_config=config, requests={"req": request})
    monkeypatch.setattr(Scheduler, "_update_after_schedule", lambda *_args: None)
    monkeypatch.setattr(Scheduler, "update_from_output", lambda *_args: None)
    _patch_scheduler_update_after_schedule()
    _patch_scheduler_update_from_output()
    old_output = SimpleNamespace(num_scheduled_tokens={"req": 1})
    Scheduler._update_after_schedule(scheduler, old_output)
    sampled = SimpleNamespace(spec_token_ids=None)
    if transition == "cancel":
        scheduler.requests.clear()
        Scheduler.update_from_output(scheduler, old_output, sampled)
        return
    if transition == "reuse_id":
        request = SimpleNamespace(is_prefill_chunk=False, next_decode_eligible_step=0)
        scheduler.requests["req"] = request
    new_output = SimpleNamespace(num_scheduled_tokens={"req": 1})
    Scheduler._update_after_schedule(scheduler, new_output)
    Scheduler.update_from_output(scheduler, old_output, sampled)
    assert request.next_decode_eligible_step > 1
    assert request._ascend_pp_sampled_output_fence is new_output
    Scheduler.update_from_output(scheduler, new_output, sampled)
    assert request.next_decode_eligible_step == 0
    assert not hasattr(request, "_ascend_pp_sampled_output_fence")
