# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import ast
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.config.compilation import CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu import cudagraph_utils as core_graph
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

from vllm_ascend.worker.v2 import aclgraph_utils as graph_mod


@pytest.fixture
def runner():
    # Exercise the actual runner selector without importing its device setup.
    path = Path(graph_mod.__file__).with_name("model_runner.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "set_pp_input_view")
    namespace = {"torch": torch, "IntermediateTensors": IntermediateTensors}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

    def make(tp):
        max_tokens = 17
        max_shard_rows = (max_tokens + tp - 1) // tp
        backing = IntermediateTensors(
            {
                "hidden_states": torch.arange(max_shard_rows * 6).reshape(max_shard_rows, 2, 3).float(),
                "residual": torch.arange(max_tokens * 3).reshape(max_tokens, 3).float(),
            }
        )
        instance = SimpleNamespace(pp_input_backing=backing, intermediate_tensors=backing)
        instance.set_pp_input_view = MethodType(namespace["set_pp_input_view"], instance)
        return instance

    return make


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
def test_received_payload_selects_each_keys_actual_rows(runner, core_copy, tp):
    instance = runner(tp)
    backing = instance.pp_input_backing
    pointers = {key: tensor.data_ptr() for key, tensor in backing.tensors.items()}
    for num_tokens in (1, 3, 17, 3, 1, 17):
        shard_rows = (num_tokens + tp - 1) // tp
        incoming = IntermediateTensors(
            {
                "hidden_states": torch.full_like(backing["hidden_states"][:shard_rows], float(num_tokens)),
                "residual": torch.full_like(backing["residual"][:num_tokens], float(num_tokens)),
            }
        )
        instance.set_pp_input_view(incoming.tensors)
        output = core_copy(instance.intermediate_tensors, incoming, num_tokens)
        for key, tensor in output.tensors.items():
            assert tensor.shape == incoming[key].shape
            assert tensor.data_ptr() == pointers[key]
            torch.testing.assert_close(tensor, incoming[key])
        assert instance.pp_input_backing is backing
        assert backing["hidden_states"].shape[0] == (17 + tp - 1) // tp
        assert backing["residual"].shape[0] == 17

        # A dummy run after a short receive must get the canonical capacity.
        instance.set_pp_input_view()
        assert instance.intermediate_tensors is backing
        dummy = core_copy(instance.intermediate_tensors, None, 17, dummy_run=True)
        assert {key: tensor.shape for key, tensor in dummy.tensors.items()} == {
            key: tensor.shape for key, tensor in backing.tensors.items()
        }


def test_tp_reconstruction_preserves_capacity_until_flat_payloads_are_restored(runner, core_copy):
    instance = runner(8)
    backing = IntermediateTensors({"hidden_states": torch.zeros(17, 4), "post": torch.zeros(17, 4, 1)})
    instance.pp_input_backing = backing
    # T=2, TP=8: both full payloads have eight elements, received as flat slices.
    payload = {key: torch.ones(1) for key in backing.tensors}
    reconstructed = {key: torch.ones_like(tensor[:2]) for key, tensor in backing.tensors.items()}
    postprocess = Mock(side_effect=lambda: payload.update(reconstructed))
    incoming = AsyncIntermediateTensors(payload, comm_postprocess=[postprocess])
    instance.set_pp_input_view()
    postprocess.assert_not_called()
    output = core_copy(instance.intermediate_tensors, incoming, 2)
    postprocess.assert_called_once()
    for key, tensor in output.tensors.items():
        assert tensor.data_ptr() == backing[key].data_ptr()
        torch.testing.assert_close(tensor, reconstructed[key])


class _Payload(nn.Module):
    def __init__(self, tp, sharded):
        super().__init__()
        self.tp = tp
        self.sharded = sharded

    def forward(self, positions, intermediate_tensors, **kwargs):
        num_tokens = positions.shape[-1]
        output = {}
        for key, tensor in intermediate_tensors.tensors.items():
            rows = (num_tokens + self.tp - 1) // self.tp if self.sharded and key == "hidden_states" else num_tokens
            assert tensor.shape[0] >= rows
            output[key] = tensor[:rows] + 1
        return IntermediateTensors(output)


def _manager(monkeypatch, instance):
    manager = graph_mod.ModelAclGraphManager.__new__(graph_mod.ModelAclGraphManager)
    state = SimpleNamespace(prepare_dummy_inputs=lambda *args: {}, attn_metadata={})
    instance.model_state = state
    instance.attn_groups = []
    manager.model_runner = instance
    manager.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1))
    manager.capture_sizes = [1, 3, 17]
    manager.is_first_pp_rank = False
    manager.is_last_pp_rank = False
    manager.intermediate_tensors = None
    manager.pp_output_backing = None
    manager.pp_output_views = {}
    manager.graphs = {}
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


@pytest.mark.parametrize("tp,sharded", [(2, False), (2, True), (8, True)])
def test_capture_replay_and_recapture_use_actual_output_views(monkeypatch, runner, tp, sharded):
    instance = runner(tp if sharded else 1)
    manager = _manager(monkeypatch, instance)
    manager.use_breakable_cg = True
    monkeypatch.setattr(graph_mod, "BreakableACLGraphWrapper", lambda model, config: model)
    monkeypatch.setattr(core_graph.BreakableCUDAGraphWrapper, "clear_all_graphs", lambda: None)
    backing = instance.pp_input_backing
    input_pointers = {key: tensor.data_ptr() for key, tensor in backing.tensors.items()}
    instance.set_pp_input_view({key: tensor[:1] for key, tensor in backing.tensors.items()})
    runtime_view = instance.intermediate_tensors
    input_buffers = SimpleNamespace(input_ids=torch.arange(17), positions=torch.arange(17), is_padding=torch.zeros(17))
    capacities = []

    def capture_driver(self, create_forward_fn, progress_bar_desc):
        # PW is captured first and can use fewer rows than the largest FULL
        # graph. Keep the real core input slices and output copy operations.
        desc = core_graph.BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 3, 1)
        forward = create_forward_fn(desc, True)
        forward(CUDAGraphMode.NONE)
        piecewise_backing = manager.pp_output_backing
        capacities.append({key: tensor.shape[0] for key, tensor in manager.pp_output_backing.tensors.items()})
        manager.pp_output_views.clear()
        forward(CUDAGraphMode.PIECEWISE)
        assert set(manager.pp_output_views) == {3}
        for num_tokens in (17, 3, 1):
            desc = core_graph.BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens, 1)
            create_forward_fn(desc, True)(CUDAGraphMode.NONE)
            create_forward_fn(desc, False)(CUDAGraphMode.NONE)
            expected_rows = {
                "hidden_states": (num_tokens + tp - 1) // tp if sharded else num_tokens,
                "residual": num_tokens,
            }
            assert {
                key: tensor.shape[0] for key, tensor in manager.intermediate_tensors.tensors.items()
            } == expected_rows
            self.graphs[desc] = object()
        assert manager.pp_output_backing is piecewise_backing
        assert instance.intermediate_tensors is backing

    monkeypatch.setattr(core_graph.CudaGraphManager, "capture", capture_driver)
    model = _Payload(tp, sharded)
    instance.set_pp_input_view()
    with model.register_forward_hook(manager._prepare_pp_output, with_kwargs=True) as handle:
        manager.capture(model, instance.model_state, input_buffers, instance.intermediate_tensors, None, [], None)
    assert not model._forward_hooks
    retained_wrapper = manager.breakable_cg_runner
    assert retained_wrapper.original_model is model
    assert not hasattr(retained_wrapper, "output_callback")
    assert capacities[0] == {"hidden_states": (3 + tp - 1) // tp if sharded else 3, "residual": 3}
    output_backing = manager.pp_output_backing
    assert output_backing["hidden_states"].shape == ((17 + tp - 1) // tp if sharded else 17, 2, 3)
    assert output_backing["residual"].shape == (17, 3)
    pointers = {key: tensor.data_ptr() for key, tensor in output_backing.tensors.items()}

    monkeypatch.setattr(graph_mod, "set_current_vllm_config", lambda *args: nullcontext())
    monkeypatch.setattr(graph_mod, "_get_graph_update_backend", lambda *args: object())
    monkeypatch.setattr(graph_mod, "use_updatable_graph", lambda *args: False)
    monkeypatch.setattr(core_graph.CudaGraphManager, "run_fullgraph", lambda *args: None)
    manager._graph_relay = lambda backend, desc, num_tokens, metadata: core_graph.ModelCudaGraphManager.run_fullgraph(
        manager, desc
    )
    for num_tokens in (3, 1, 17):
        output = manager.run_fullgraph(SimpleNamespace(num_tokens=num_tokens))
        for key, tensor in output.tensors.items():
            rows = (num_tokens + tp - 1) // tp if sharded and key == "hidden_states" else num_tokens
            assert tensor.shape[0] == rows
            assert tensor.data_ptr() == pointers[key]
            torch.testing.assert_close(tensor, backing[key][:rows] + 1)

    manager._graphs_captured = True
    manager._capture_descs = {CUDAGraphMode.FULL: [SimpleNamespace(num_tokens=17)]}
    manager.release_graphs()
    assert not manager.graphs
    assert not manager._graphs_captured
    assert manager.needs_capture()
    instance.set_pp_input_view(runtime_view.tensors)
    instance.set_pp_input_view()
    manager.pp_output_views.clear()
    with model.register_forward_hook(manager._prepare_pp_output, with_kwargs=True) as recapture_handle:
        assert recapture_handle.id != handle.id
        manager.capture(model, instance.model_state, input_buffers, instance.intermediate_tensors, None, [], None)
    assert not model._forward_hooks
    assert set(manager.pp_output_views) == {1, 3, 17}
    assert manager.breakable_cg_runner is retained_wrapper
    assert manager.pp_output_backing is output_backing
    assert {key: tensor.data_ptr() for key, tensor in output_backing.tensors.items()} == pointers
    assert {key: tensor.data_ptr() for key, tensor in backing.tensors.items()} == input_pointers


def test_output_storage_cannot_grow_after_full_capture(monkeypatch, runner):
    manager = _manager(monkeypatch, runner(2))
    manager._prepare_pp_output(
        None, (), {"positions": torch.arange(3)}, IntermediateTensors({"hidden_states": torch.zeros(2, 3)})
    )
    manager.graphs["captured"] = object()
    with pytest.raises(RuntimeError, match="cannot grow"):
        manager._prepare_pp_output(
            None, (), {"positions": torch.arange(5)}, IntermediateTensors({"hidden_states": torch.zeros(3, 3)})
        )


def test_capture_failure_finishes_metadata_and_passes_canonical_inputs(monkeypatch, runner):
    instance = runner(2)
    manager = _manager(monkeypatch, instance)
    instance.model_state.device_metadata = SimpleNamespace(finish=Mock())
    instance.set_pp_input_view({key: tensor[:1] for key, tensor in instance.pp_input_backing.tensors.items()})
    runtime_view = instance.intermediate_tensors

    def fail_capture(self, model, state, buffers, intermediate_tensors, *args, **kwargs):
        assert intermediate_tensors is instance.pp_input_backing
        raise RuntimeError("capture failed")

    monkeypatch.setattr(core_graph.ModelCudaGraphManager, "capture", fail_capture)
    model = _Payload(2, True)
    instance.set_pp_input_view()
    with (
        pytest.raises(RuntimeError, match="capture failed"),
        model.register_forward_hook(manager._prepare_pp_output, with_kwargs=True),
    ):
        manager.capture(model, instance.model_state, None, instance.intermediate_tensors, None, [], None)
    assert instance.intermediate_tensors is instance.pp_input_backing
    assert instance.intermediate_tensors is not runtime_view
    assert not model._forward_hooks
    instance.model_state.device_metadata.finish.assert_called_once()


def test_output_hook_uses_last_positions_axis_and_finishes_on_failure(monkeypatch, runner):
    instance = runner(2)
    manager = _manager(monkeypatch, instance)
    metadata = SimpleNamespace(begin_forward=Mock(), finish=Mock())
    payload = _Payload(2, True)
    model = graph_mod.ModelWithContext(payload, device_metadata=metadata)
    with (
        pytest.raises(RuntimeError, match="cannot grow"),
        payload.register_forward_hook(manager._prepare_pp_output, with_kwargs=True),
    ):
        output = model(positions=torch.zeros(2, 3), intermediate_tensors=instance.pp_input_backing)
        assert isinstance(output, IntermediateTensors)
        assert set(manager.pp_output_views) == {3}
        manager.graphs["captured"] = object()
        model(positions=torch.zeros(3, 17), intermediate_tensors=instance.pp_input_backing)
    assert not payload._forward_hooks
    assert metadata.begin_forward.call_count == 2
    assert metadata.finish.call_count == 2
