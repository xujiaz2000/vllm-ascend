# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.sequence import IntermediateTensors

from vllm_ascend.worker.v2 import pp_transport
from vllm_ascend.worker.v2.pp_transport import resolve_spec_pp_support


@pytest.mark.parametrize(
    "indexer_type,use_index_cache,pattern,expected",
    [
        ("full", True, "FFSF", True),
        ("full", True, "FFFF", False),
        ("full", False, "FFSF", False),
        ("shared", False, "FFFF", True),
        ("shared", True, "FFFF", True),
        ("shared", True, "FFSF", True),
    ],
)
def test_topk_boundary_combines_indexshare_and_indexcache(indexer_type, use_index_cache, pattern, expected):
    config = SimpleNamespace(
        num_hidden_layers=4,
        indexer_types=["full", "shared", indexer_type, "shared"],
        use_index_cache=use_index_cache,
        index_topk_pattern=pattern,
    )
    assert pp_transport.pp_stage_requires_topk_indices(config, 2) is expected
    assert not pp_transport.pp_stage_requires_topk_indices(config, 0)
    assert not pp_transport.pp_stage_requires_topk_indices(config, config.num_hidden_layers)


@pytest.mark.parametrize(
    "method,architecture,needs_aux",
    [
        ("mtp", "Qwen3_5ForConditionalGeneration", False),
        ("eagle3", "MiniMaxM3SparseForCausalLM", True),
        ("eagle3", "MiniMaxM3SparseForConditionalGeneration", True),
        ("dspark", "DeepseekV4ForCausalLM", True),
        ("dspark", "GlmMoeDsaForCausalLM", True),
        ("mtp", "Glm5NextForCausalLM", False),
        ("mtp", "Glm5NextForConditionalGeneration", False),
        ("dflash", "Glm5NextForCausalLM", True),
        ("dflash", "Glm5NextForConditionalGeneration", True),
    ],
)
@pytest.mark.parametrize("pp_size", [1, 2])
def test_resolve_spec_pp_support(method, architecture, needs_aux, pp_size):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        model_config=SimpleNamespace(architecture=architecture),
        speculative_config=SimpleNamespace(method=method),
    )
    support = resolve_spec_pp_support(config)
    if pp_size == 1:
        assert support is None
    else:
        assert support.needs_aux_hidden_states is needs_aux


@pytest.mark.parametrize("method", [None, "ngram", "eagle3"])
def test_unregistered_spec_pp_support(method):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        model_config=SimpleNamespace(architecture="OtherModel"),
        speculative_config=SimpleNamespace(method=method) if method else None,
    )
    assert resolve_spec_pp_support(config) is None


def test_aux_buffer_includes_stage_boundary():
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=4), start_layer=2, aux_hidden_state_layers=(0, 2, 4))
    factory = pp_transport.make_empty_intermediate_tensors(
        model, lambda batch, dtype, device: IntermediateTensors({"hidden_states": torch.zeros(batch, 4)})
    )
    tensors = factory(3, torch.float32, torch.device("cpu"))
    aux = pp_transport.get_pp_transport_tensors(tensors, pp_transport.PPTransportDataType.AUX_HIDDEN_STATES)
    assert len(aux) == 2
    assert all(t.shape == (3, 4) for t in aux)


def _make_config(
    architecture: str,
    *,
    method: str = "dspark",
    pipeline_parallel_size: int = 2,
):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method=method),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
        ),
        model_config=SimpleNamespace(architecture=architecture),
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "KimiLinearForCausalLM",
        "KimiK3ForCausalLM",
        "KimiK3ForConditionalGeneration",
    ],
)
def test_kimi_k3_dspark_pp_supports_all_target_aliases(architecture):
    support = resolve_spec_pp_support(_make_config(architecture))

    assert support is not None
    assert support.needs_aux_hidden_states


@pytest.mark.parametrize(
    "config",
    [
        _make_config("KimiLinearForCausalLM", pipeline_parallel_size=1),
        _make_config("UnsupportedForPP"),
        _make_config("KimiLinearForCausalLM", method="eagle"),
    ],
)
def test_kimi_k3_dspark_pp_support_stays_scoped(config):
    assert resolve_spec_pp_support(config) is None


@pytest.mark.parametrize("method", ["mtp", "dflash"])
@pytest.mark.parametrize(
    "architecture,pp_size,expected",
    [
        ("Glm5NextForCausalLM", 2, True),
        ("Glm5NextForConditionalGeneration", 2, True),
        ("Glm5NextForCausalLM", 1, False),
        ("OtherModel", 2, False),
    ],
)
def test_sampled_output_fence_stays_scoped(method, architecture, pp_size, expected):
    config = _make_config(architecture, method=method, pipeline_parallel_size=pp_size)
    assert pp_transport.requires_pp_sampled_output_fence(config) is expected


@pytest.mark.parametrize(
    "architecture,method,pp_size,use_v2,async_scheduling,rejected",
    [
        ("Glm5NextForCausalLM", "mtp", 2, True, True, True),
        ("Glm5NextForConditionalGeneration", "mtp", 2, True, True, True),
        ("Glm5NextForCausalLM", "mtp", 3, True, True, True),
        ("Glm5NextForCausalLM", "mtp", 2, True, False, False),
        ("Glm5NextForConditionalGeneration", "mtp", 2, True, False, False),
        ("Glm5NextForCausalLM", "mtp", 1, True, True, False),
        ("Glm5NextForCausalLM", "mtp", 2, False, True, False),
        ("Glm5NextForCausalLM", "dflash", 2, True, True, False),
        ("Glm5NextForConditionalGeneration", "dflash", 2, True, True, False),
        ("Qwen3_5ForConditionalGeneration", "mtp", 2, True, True, False),
        ("OtherModel", "mtp", 2, True, True, False),
        ("Glm5NextForCausalLM", None, 2, True, True, False),
    ],
)
def test_unvalidated_async_mtp_pp_fails_before_model_loading(
    architecture, method, pp_size, use_v2, async_scheduling, rejected
):
    config = SimpleNamespace(
        use_v2_model_runner=use_v2,
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        model_config=SimpleNamespace(architecture=architecture),
        speculative_config=SimpleNamespace(method=method) if method else None,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )
    if rejected:
        with pytest.raises(ValueError, match="requires --no-async-scheduling"):
            resolve_spec_pp_support(config)
    else:
        resolve_spec_pp_support(config)
