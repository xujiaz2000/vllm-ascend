# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the DSV4 PP boundary with real forward, factory and SP helpers."""

import __future__

import ast
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.ut.models.test_kimi_k3_pp_sp import ROOT, Collectives, IntermediateTensors, load_definitions
from vllm_ascend.worker.v2 import pp_transport

MODEL_PATH = "vllm_ascend/models/deepseek_v4/model.py"
HIDDEN_SIZE = 3
HC_MULT = 2
AUX_LAYERS = (1, 3, 5)


@pytest.fixture
def runtime(monkeypatch):
    context = threading.local()
    namespace = {
        "torch": torch,
        "F": torch.nn.functional,
        "BaseModel": nn.Module,
        "islice": islice,
        "IntermediateTensors": IntermediateTensors,
        "envs": SimpleNamespace(VLLM_MOE_SKIP_PADDING=True),
        "is_forward_context_available": lambda: True,
        "get_forward_context": lambda: context.forward,
        "get_pp_group": lambda: context.pp,
        "get_tensor_model_parallel_world_size": lambda: context.tp,
        "get_tensor_model_parallel_rank": lambda: context.rank,
        "get_pp_transport_tensors": pp_transport.get_pp_transport_tensors,
        "add_pp_transport_tensors": pp_transport.add_pp_transport_tensors,
        "PPTransportDataType": pp_transport.PPTransportDataType,
    }
    load_definitions(
        "vllm_ascend/models/common/ops/sequence_parallel.py",
        {"sp_shard", "sp_padding_mask", "_ascend_sp_shard_impl", "_ascend_sp_padding_mask_impl"},
        namespace,
    )
    for name in ("ascend_sp_shard_impl", "ascend_sp_padding_mask_impl"):
        monkeypatch.setattr(torch.ops.vllm, name, namespace[f"_{name}"], raising=False)
    load_definitions(
        MODEL_PATH,
        {"DeepseekV4Model", "AscendDeepseekV4ForCausalLM"},
        namespace,
        bases={"DeepseekV4Model": "BaseModel", "AscendDeepseekV4ForCausalLM": "BaseModel"},
        methods={"DeepseekV4Model": {"forward", "hc_head"}, "AscendDeepseekV4ForCausalLM": {"forward"}},
    )
    return namespace, context


class _Layer(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, positions, hidden, residual, scaling, input_ids=None):
        # Distinct token and residual-stream values detect wrong-rank copies.
        stream_bias = torch.arange(HC_MULT, dtype=hidden.dtype).view(1, HC_MULT, 1) / 100
        return hidden * 1.1 + (self.layer_idx + 1) / 10 + stream_bias, residual


def _model(namespace, start, end, *, sp):
    model = namespace["DeepseekV4Model"]()
    model.start_layer, model.end_layer = start, end
    model.layers = nn.ModuleList([_Layer(i) for i in range(6)])
    model.config = SimpleNamespace(hidden_size=HIDDEN_SIZE)
    model.hc_mult = HC_MULT
    model.hc_eps = 1e-5
    model.hc_norm = nn.Identity()
    model.norm = nn.Identity()
    model.hc_head_fn = torch.arange(HC_MULT * HC_MULT * HIDDEN_SIZE).reshape(HC_MULT, -1).float() / 100
    model.hc_head_scale = torch.ones(1)
    model.hc_head_base = torch.zeros(HC_MULT)
    model.aux_hidden_state_layers = AUX_LAYERS
    model.use_sequence_parallel_moe = sp
    model._mtp_hidden_buffer = torch.full((24, HC_MULT * HIDDEN_SIZE), -999.0)

    # Compile the unchanged nested production factory without constructing NPU layers.
    tree = ast.parse((ROOT / MODEL_PATH).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if getattr(node, "name", None) == "DeepseekV4Model")
    init = next(node for node in cls.body if getattr(node, "name", None) == "__init__")
    factory = next(node for node in init.body if getattr(node, "name", None) == "make_empty_intermediate_tensors")
    factory_namespace = {**namespace, "self": model, "config": model.config}
    exec(
        compile(ast.Module(body=[factory], type_ignores=[]), MODEL_PATH, "exec", __future__.annotations.compiler_flag),
        factory_namespace,
    )
    model.make_empty_intermediate_tensors = pp_transport.make_empty_intermediate_tensors(
        model, factory_namespace["make_empty_intermediate_tensors"]
    )
    return model


@pytest.mark.parametrize("tp", [1, 2, 8])
@pytest.mark.parametrize("cuts", [(0, 6), (0, 3, 6), (0, 2, 4, 6)])
def test_dsv4_pipeline_keeps_all_payloads_sharded_until_last_stage(runtime, tp, cuts):
    namespace, context = runtime
    collectives = Collectives(tp, context)
    num_stages = len(cuts) - 1
    gather_stages = [[] for _ in range(tp)]
    shard_stages = [[] for _ in range(tp)]
    sp_shard = namespace["sp_shard"]

    def gather(value):
        gather_stages[context.rank].append(context.stage)
        return collectives.exchange(value)

    def shard(value):
        shard_stages[context.rank].append(context.stage)
        return sp_shard(value)

    namespace["sp_all_gather"] = gather
    namespace["sp_shard"] = shard
    context.tp, context.rank = 1, 0
    context.pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    reference = _model(namespace, 0, 6, sp=False)
    reference_wrapper = namespace["AscendDeepseekV4ForCausalLM"]()
    reference_wrapper.model = reference
    token_sizes = (1, 3, 17, 3, 1, 17)
    expected = {}
    expected_mtp = {}
    for num_tokens in set(token_sizes):
        inputs = torch.arange(num_tokens * HIDDEN_SIZE).reshape(num_tokens, HIDDEN_SIZE).float() / 10
        expected[num_tokens] = reference_wrapper(None, torch.arange(num_tokens), None, inputs_embeds=inputs)
        expected_mtp[num_tokens] = reference._mtp_hidden_buffer[:num_tokens].clone()

    def run_rank(rank):
        context.tp, context.rank = tp, rank
        models = [_model(namespace, start, end, sp=True) for start, end in zip(cuts, cuts[1:])]
        wrappers = [namespace["AscendDeepseekV4ForCausalLM"]() for _ in models]
        for wrapper, model in zip(wrappers, models):
            wrapper.model = model
        # Allocate the maximum once, then alternate grow/shrink active views.
        backing = [model.make_empty_intermediate_tensors(max(token_sizes), torch.float32, "cpu") for model in models]
        pointers = [{key: tensor.data_ptr() for key, tensor in bank.items()} for bank in backing]
        for num_tokens in token_sizes:
            inputs = torch.arange(num_tokens * HIDDEN_SIZE).reshape(num_tokens, HIDDEN_SIZE).float() / 10
            positions = torch.arange(num_tokens)
            shard_rows = (num_tokens + tp - 1) // tp
            intermediate = None
            for stage, model in enumerate(models):
                context.stage = stage
                context.pp = SimpleNamespace(is_first_rank=stage == 0, is_last_rank=stage == num_stages - 1)
                context.forward = SimpleNamespace(is_padding=torch.zeros(num_tokens, dtype=torch.bool))
                if intermediate is not None:
                    bank = backing[stage]
                    assert intermediate.tensors.keys() == bank.tensors.keys()
                    active = bank[:shard_rows]
                    for key, tensor in intermediate.items():
                        assert tensor.shape[0] == shard_rows
                        active[key].copy_(tensor)
                        assert active[key].data_ptr() == pointers[stage][key]
                    intermediate = active
                output = wrappers[stage](
                    positions if stage == 0 else None,
                    positions,
                    intermediate,
                    inputs_embeds=inputs if stage == 0 else None,
                )
                padding = torch.arange(shard_rows) + rank * shard_rows >= num_tokens
                torch.testing.assert_close(context.forward.is_padding, padding)
                if stage < num_stages - 1:
                    assert output["hidden_states"].shape == (shard_rows, HC_MULT, HIDDEN_SIZE)
                    aux = pp_transport.get_pp_transport_tensors(
                        output, pp_transport.PPTransportDataType.AUX_HIDDEN_STATES
                    )
                    assert len(aux) == sum(layer <= model.end_layer for layer in AUX_LAYERS)
                    assert all(t.shape == (shard_rows, HIDDEN_SIZE) for t in aux)
                    assert all(torch.all(t[padding] == 0) for t in output.tensors.values())
                    intermediate = output
                else:
                    hidden, aux = output
                    expected_hidden, expected_aux = expected[num_tokens]
                    torch.testing.assert_close(hidden, expected_hidden)
                    for actual, wanted in zip(aux, expected_aux, strict=True):
                        torch.testing.assert_close(actual, wanted)
                    torch.testing.assert_close(model._mtp_hidden_buffer[:num_tokens], expected_mtp[num_tokens])
        assert {stage for stage in shard_stages[rank]} == {0}
        assert len(shard_stages[rank]) == 2 * len(token_sizes)
        assert gather_stages[rank] == [num_stages - 1] * ((1 + len(AUX_LAYERS)) * len(token_sizes))

    with ThreadPoolExecutor(max_workers=tp) as pool:
        list(pool.map(run_rank, range(tp)))


@pytest.mark.parametrize("tp", [2, 8])
@pytest.mark.parametrize("num_tokens", [1, 3, 17])
def test_outer_forward_normalizes_dummy_inputs_before_compilation(runtime, tp, num_tokens):
    namespace, context = runtime
    context.tp = tp
    rows = (num_tokens + tp - 1) // tp
    canonical = IntermediateTensors({"hidden_states": torch.ones(17, 2, 3), "aux": torch.ones(17, 3)})

    class CompiledModel(nn.Module):
        use_sequence_parallel_moe = True

        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            assert all(tensor.shape[0] == rows for tensor in intermediate_tensors.tensors.values())
            return IntermediateTensors({key: tensor.clone() for key, tensor in intermediate_tensors.items()})

    wrapper = namespace["AscendDeepseekV4ForCausalLM"]()
    wrapper.model = CompiledModel()
    for rank in range(tp):
        context.rank = rank
        output = wrapper(None, torch.arange(num_tokens), canonical[:num_tokens])
        valid = max(0, min(rows, num_tokens - rank * rows))
        for tensor in output.tensors.values():
            assert torch.all(tensor[:valid] == 1)
            assert torch.all(tensor[valid:] == 0)
