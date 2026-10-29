# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.config.compilation import CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu import cudagraph_utils as core_graph
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from vllm_ascend.worker.v2 import aclgraph_utils as graph_mod
from vllm_ascend.worker.v2 import pp_transport
from vllm_ascend.worker.v2.model_states.default import AscendModelState


def _config(tp=2):
    return SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(architecture="DeepseekV4ForCausalLM"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=3,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_sequence_parallel_moe=True,
        ),
    )


@pytest.fixture(autouse=True)
def _sp_enabled(monkeypatch):
    monkeypatch.setattr(pp_transport, "enable_sp", lambda config: config.parallel_config.use_sequence_parallel_moe)


@pytest.mark.parametrize("architecture", ["DeepseekV4ForCausalLM", "DeepseekV4ForConditionalGeneration"])
@pytest.mark.parametrize(
    "owner,field,value",
    [
        (None, None, None),
        ("config", "use_v2_model_runner", False),
        ("parallel", "use_sequence_parallel_moe", False),
        ("parallel", "pipeline_parallel_size", 1),
        ("parallel", "tensor_parallel_size", 1),
        ("parallel", "prefill_context_parallel_size", 2),
        ("parallel", "decode_context_parallel_size", 2),
        ("model", "architecture", "DeepseekV41ForCausalLM"),
        ("model", "architecture", "KimiK3ForCausalLM"),
        ("model", "architecture", "GlmMoeDsaForCausalLM"),
    ],
)
def test_dsv4_pp_sp_gate(architecture, owner, field, value):
    config = _config()
    config.model_config.architecture = architecture
    if owner is not None:
        target = {"config": config, "parallel": config.parallel_config, "model": config.model_config}[owner]
        setattr(target, field, value)
    assert pp_transport.use_dsv4_pp_sp(config) is (owner is None)


def _state(monkeypatch, tp):
    state = AscendModelState.__new__(AscendModelState)
    state.vllm_config = _config(tp)
    state.prepare_engram_inputs = lambda *args: {}
    state.prepare_engram_dummy_inputs = lambda *args: {}
    monkeypatch.setattr(DefaultModelState, "prepare_dummy_inputs", lambda *args: {})
    max_tokens = 17
    max_shard_rows = (max_tokens + tp - 1) // tp
    tensors = {
        "hidden_states": torch.arange(max_shard_rows * 6).reshape(max_shard_rows, 2, 3).float(),
        "pp_transport_aux_hidden_states_0": torch.arange(max_tokens * 3).reshape(max_tokens, 3).float(),
    }
    state.pp_input_backing = IntermediateTensors(tensors)
    state.pp_input_tensors = state.pp_input_backing[:1]
    return state


@pytest.fixture
def core_copy():
    # Evaluate the real core [:T].copy_ expression, avoiding runner construction.
    path = Path(core_graph.__file__).with_name("model_runner.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    expression = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "new_tensors" for target in node.targets)
        and isinstance(node.value, ast.DictComp)
    )
    code = compile(ast.Expression(expression), str(path), "eval")

    def copy(active, incoming, num_tokens, *, dummy_run=False):
        return IntermediateTensors(
            eval(
                code,
                {
                    "self": SimpleNamespace(intermediate_tensors=active),
                    "intermediate_tensors": incoming,
                    "n": num_tokens,
                    "dummy_run": dummy_run,
                },
            )
        )

    return copy


@pytest.mark.parametrize("tp", [2, 8])
def test_prepare_inputs_activates_uniform_shards_before_core_copy(monkeypatch, core_copy, tp):
    state = _state(monkeypatch, tp)
    active = state.pp_input_tensors
    backing = state.pp_input_backing
    pointers = {key: tensor.data_ptr() for key, tensor in backing.tensors.items()}

    def prepare_parent(self, input_batch, req_states):
        assert self.pp_num_tokens == input_batch.num_tokens_after_padding
        shard_rows = (self.pp_num_tokens + tp - 1) // tp
        assert all(tensor.shape[0] == shard_rows for tensor in self.pp_input_tensors.tensors.values())
        return {"prepared": True}

    monkeypatch.setattr(DefaultModelState, "prepare_inputs", prepare_parent)
    for num_tokens in (1, 3, 17, 3, 1, 17):
        assert state.prepare_inputs(SimpleNamespace(num_tokens_after_padding=num_tokens), None) == {"prepared": True}
        assert state.pp_input_tensors is active
        shard_rows = (num_tokens + tp - 1) // tp
        incoming = IntermediateTensors(
            {key: torch.full_like(tensor[:shard_rows], float(num_tokens)) for key, tensor in backing.tensors.items()}
        )
        output = core_copy(active, incoming, num_tokens)
        for key, tensor in output.tensors.items():
            assert tensor.shape[0] == shard_rows
            assert tensor.data_ptr() == pointers[key]
            torch.testing.assert_close(tensor, incoming[key])
        assert backing["hidden_states"].shape[0] == (17 + tp - 1) // tp
        assert backing["pp_transport_aux_hidden_states_0"].shape[0] == 17


@pytest.mark.parametrize("tp,num_tokens", [(2, 1), (2, 3), (8, 1), (8, 3), (8, 17)])
def test_padding_clear_applies_to_hidden_and_aux(tp, num_tokens):
    shard_rows = (num_tokens + tp - 1) // tp
    for rank in range(tp):
        tensors = {"hidden_states": torch.ones(shard_rows, 2, 3), "aux": torch.ones(shard_rows, 3)}
        pp_transport.zero_pp_shard_padding(tensors, num_tokens, tp, rank)
        valid = max(0, min(shard_rows, num_tokens - rank * shard_rows))
        for tensor in tensors.values():
            assert torch.all(tensor[:valid] == 1)
            assert torch.all(tensor[valid:] == 0)


class _Payload(nn.Module):
    def __init__(self, tp):
        super().__init__()
        self.tp = tp

    def forward(self, positions, intermediate_tensors, **kwargs):
        shard_rows = (positions.shape[0] + self.tp - 1) // self.tp
        assert all(tensor.shape[0] == shard_rows for tensor in intermediate_tensors.tensors.values())
        return IntermediateTensors({key: tensor + 1 for key, tensor in intermediate_tensors.tensors.items()})


def _manager(monkeypatch, state):
    manager = graph_mod.ModelAclGraphManager.__new__(graph_mod.ModelAclGraphManager)
    manager.model_runner = SimpleNamespace(model_state=state, attn_groups=[])
    manager.vllm_config = state.vllm_config
    manager.capture_sizes = [1, 3, 17]
    manager.is_first_pp_rank = False
    manager.is_last_pp_rank = False
    manager.intermediate_tensors = None
    manager.update_stream = object()
    manager.use_breakable_cg = False
    manager.cudagraph_mode = CUDAGraphMode.FULL
    manager.max_num_reqs = 4
    manager.breakable_cg_runner = None
    monkeypatch.setattr(graph_mod, "communicator_switch", nullcontext)
    monkeypatch.setattr(graph_mod, "_EXTRA_CTX", SimpleNamespace())
    monkeypatch.setattr(
        graph_mod, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)
    monkeypatch.setattr(core_graph, "set_forward_context", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(core_graph, "prepare_inputs_to_capture", lambda *args, **kwargs: ({}, None))
    monkeypatch.setattr(core_graph, "requires_raw_input_tokens", lambda model: False)
    return manager


@pytest.mark.parametrize("tp", [2, 8])
def test_capture_replay_release_keep_views_and_storage_stable(monkeypatch, tp):
    state = _state(monkeypatch, tp)
    manager = _manager(monkeypatch, state)
    previous_layout = Mock()
    state.pp_capture_layout = previous_layout
    runtime_active = state.pp_input_tensors
    runtime_dictionary = runtime_active.tensors
    input_pointers = {key: tensor.data_ptr() for key, tensor in state.pp_input_backing.tensors.items()}
    input_buffers = SimpleNamespace(input_ids=torch.arange(17), positions=torch.arange(17), is_padding=torch.zeros(17))

    def capture_driver(self, create_forward_fn, progress_bar_desc):
        # Keep the real core input slices, output copies and callback ordering.
        for num_tokens in (17, 3, 1, 17):
            desc = core_graph.BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens, 1)
            create_forward_fn(desc, True)(CUDAGraphMode.FULL)
            assert state.pp_input_tensors is runtime_active
            assert runtime_active.tensors is runtime_dictionary
            assert all(tensor.shape[0] == 1 for tensor in runtime_active.tensors.values())

    monkeypatch.setattr(core_graph.CudaGraphManager, "capture", capture_driver)
    model = _Payload(tp)
    manager.capture(model, state, input_buffers, runtime_active, None, [], None)
    assert state.pp_capture_layout is previous_layout
    previous_layout.assert_not_called()
    backing = manager.pp_output_backing
    assert backing["hidden_states"].shape == ((17 + tp - 1) // tp, 2, 3)
    assert backing["pp_transport_aux_hidden_states_0"].shape == ((17 + tp - 1) // tp, 3)
    pointers = {key: tensor.data_ptr() for key, tensor in backing.tensors.items()}

    state.attn_metadata = {}
    monkeypatch.setattr(graph_mod, "set_current_vllm_config", lambda *args: nullcontext())
    monkeypatch.setattr(graph_mod, "_get_graph_update_backend", lambda *args: object())
    monkeypatch.setattr(graph_mod, "use_updatable_graph", lambda *args: False)
    monkeypatch.setattr(core_graph.CudaGraphManager, "run_fullgraph", lambda *args: None)
    manager._graph_relay = lambda backend, desc, num_tokens, metadata: core_graph.ModelCudaGraphManager.run_fullgraph(
        manager, desc
    )
    for num_tokens in (3, 1, 17):
        output = manager.run_fullgraph(SimpleNamespace(num_tokens=num_tokens))
        shard_rows = (num_tokens + tp - 1) // tp
        for key, tensor in output.tensors.items():
            assert tensor.shape[0] == shard_rows
            assert tensor.data_ptr() == pointers[key]
            torch.testing.assert_close(tensor, state.pp_input_backing[key][:shard_rows] + 1)

    manager.graphs = {"captured": object()}
    manager._graphs_captured = True
    manager._capture_descs = {CUDAGraphMode.FULL: [SimpleNamespace(num_tokens=17)]}
    manager.release_graphs()
    assert not manager.graphs
    assert not manager._graphs_captured
    assert manager.needs_capture()
    assert manager.pp_output_backing is backing
    manager.capture(model, state, input_buffers, runtime_active, None, [], None)
    assert {key: tensor.data_ptr() for key, tensor in backing.tensors.items()} == pointers
    assert {key: tensor.data_ptr() for key, tensor in state.pp_input_backing.tensors.items()} == input_pointers


def test_capture_failure_restores_callback_and_runtime_layout(monkeypatch):
    state = _state(monkeypatch, 2)
    manager = _manager(monkeypatch, state)
    previous_layout = Mock()
    state.pp_capture_layout = previous_layout
    state.device_metadata = SimpleNamespace(finish=Mock())
    dictionary = state.pp_input_tensors.tensors

    def fail_capture(*args, **kwargs):
        state.prepare_dummy_inputs(1, 3)
        raise RuntimeError("capture failed")

    monkeypatch.setattr(core_graph.ModelCudaGraphManager, "capture", fail_capture)
    with pytest.raises(RuntimeError, match="capture failed"):
        manager.capture(_Payload(2), state, None, state.pp_input_tensors, None, [], None)
    assert state.pp_capture_layout is previous_layout
    assert state.pp_input_tensors.tensors is dictionary
    state.device_metadata.finish.assert_called_once()
