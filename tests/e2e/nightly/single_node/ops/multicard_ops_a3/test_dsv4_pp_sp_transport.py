# SPDX-License-Identifier: Apache-2.0
"""Check DSV4's production PP boundaries with real P2P and TP collectives.

Run explicitly with 2 * TP processes and --device npu --tp-size 2 (or 8).
CPU/Gloo is a control-flow check. AST loading avoids model weights and optional
kernel dependencies; deterministic decoder doubles exclude attention numerics
and real ACL graph execution from this component test.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import pickle
from collections import deque, namedtuple
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

HIDDEN_SIZE = 4
HC_STREAMS = 2
AUX_KEY = "pp_transport_aux_hidden_states_0"
TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


def _load(path, names, namespace, class_name=None):
    body = ast.parse(path.read_text(encoding="utf-8")).body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    nodes = [copy.deepcopy(node) for node in body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names), (path, names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def _production(repo, core):
    namespace = {
        "torch": torch,
        "pickle": pickle,
        "deque": deque,
        "dataclass": dataclass,
        "TensorMetadata": TensorMetadata,
        "Enum": Enum,
        "islice": islice,
        "F": torch.nn.functional,
        "_PP_TRANSPORT_PREFIX": "pp_transport",
    }
    _load(core / "vllm/sequence.py", ("IntermediateTensors",), namespace)
    parallel = core / "vllm/distributed/parallel_state.py"
    _load(parallel, ("_RetainedHandle", "_split_tensor_dict"), namespace)
    methods = (
        "isend_object",
        "recv_object",
        "_should_use_all_gather",
        "_reap_completed_isends",
        "isend_tensor_dict",
        "irecv_tensor_dict",
    )
    _load(parallel, methods, namespace, "GroupCoordinator")
    coordinator = type("ProductionP2P", (), {name: namespace[name] for name in methods})
    _load(
        repo / "vllm_ascend/worker/v2/pp_transport.py",
        (
            "zero_pp_shard_padding",
            "PPTransportDataType",
            "_get_transport_key_prefix",
            "get_pp_transport_tensors",
            "add_pp_transport_tensors",
        ),
        namespace,
    )
    _load(
        repo / "vllm_ascend/models/common/ops/sequence_parallel.py",
        ("_ascend_sp_shard_impl",),
        namespace,
    )
    _load(repo / "vllm_ascend/models/deepseek_v4/model.py", ("forward",), namespace, "DeepseekV4Model")
    return coordinator, namespace


def _groups(rank, backend, coordinator_type, tp):
    pairs, cpu_pairs, tp_groups = {}, {}, {}
    for lane in range(tp):
        ranks = [lane, lane + tp]
        cpu_pairs[lane] = dist.new_group(ranks, backend="gloo")
        pairs[lane] = dist.new_group(ranks, backend=backend) if backend != "gloo" else cpu_pairs[lane]
    for stage in range(2):
        tp_groups[stage] = dist.new_group(list(range(stage * tp, (stage + 1) * tp)), backend=backend)
    lane = rank % tp
    coordinator = coordinator_type()
    coordinator.world_size, coordinator.rank_in_group = 2, rank // tp
    coordinator.ranks = [lane, lane + tp]
    coordinator.cpu_group, coordinator.device_group = cpu_pairs[lane], pairs[lane]
    coordinator.use_cpu_custom_send_recv = False
    coordinator.device_communicator = None
    return coordinator, tp_groups[rank // tp]


def _case(namespace, coordinator, tp_group, rank, tp, tokens, device, backing):
    lane, stage = rank % tp, rank // tp
    rows = (tokens + tp - 1) // tp
    valid = max(0, min(rows, tokens - lane * rows))
    base = torch.arange(tokens * HIDDEN_SIZE, device=device, dtype=torch.float32).reshape(tokens, HIDDEN_SIZE)
    owners = torch.arange(tokens, device=device) // rows
    wanted = base + (owners[:, None] + 1) * 1000 + 50
    gathers = []

    def gather(tensor):
        gathers.append(tuple(tensor.shape))
        parts = [torch.empty_like(tensor) for _ in range(tp)]
        dist.all_gather(parts, tensor.contiguous(), group=tp_group)
        return torch.cat(parts)

    namespace.update(
        {
            "get_pp_group": lambda: SimpleNamespace(is_first_rank=stage == 0, is_last_rank=stage == 1),
            "get_tensor_model_parallel_rank": lambda: lane,
            "get_tensor_model_parallel_world_size": lambda: tp,
            "envs": SimpleNamespace(VLLM_MOE_SKIP_PADDING=False),
            "is_forward_context_available": lambda: False,
            "sp_shard": namespace["_ascend_sp_shard_impl"],
            "sp_all_gather": gather,
        }
    )

    def decoder(positions, hidden, residual, scaling, input_ids=None):
        assert hidden.shape == (rows, HC_STREAMS, HIDDEN_SIZE)
        if stage == 0:
            offsets = torch.arange(HC_STREAMS, device=device).view(1, HC_STREAMS, 1) * 100
            return hidden + offsets + 1000 * (lane + 1), None
        assert not hidden[valid:].count_nonzero()
        torch.testing.assert_close(hidden[:valid].mean(1), wanted[lane * rows : lane * rows + valid])
        return hidden + 10, None

    model = SimpleNamespace(
        _pp_sp=True,
        use_sequence_parallel_moe=True,
        hc_mult=HC_STREAMS,
        config=SimpleNamespace(),
        start_layer=stage,
        end_layer=stage + 1,
        layers=[None, None],
        aux_hidden_state_layers=(1, 2),
        _mtp_hidden_buffer=None,
        hc_head=lambda hidden, *args: hidden.mean(1),
        hc_head_fn=None,
        hc_head_scale=None,
        hc_head_base=None,
        norm=lambda hidden: hidden,
    )

    class Layer:
        layer_idx = stage
        __call__ = staticmethod(decoder)

    model.layers[stage] = Layer()
    positions = torch.arange(tokens, device=device)
    packets = []

    def record(original, tensor, *args, **kwargs):
        if tensor.dtype == torch.float32:
            assert kwargs["dst" if stage == 0 else "src"] == (lane + tp if stage == 0 else lane)
            packets.append((tuple(tensor.shape), tensor.numel() * tensor.element_size()))
        return original(tensor, *args, **kwargs)

    if stage == 0:
        output = namespace["forward"](model, None, positions, None, inputs_embeds=base)
        assert not gathers, "The sending stage must not gather hidden or aux at its boundary"
        namespace["zero_pp_shard_padding"](output.tensors, tokens, tp, lane)
        for tensor in output.tensors.values():
            assert tensor.shape[0] == rows
            assert not tensor[valid:].count_nonzero()
        send = dist.isend
        with patch.object(dist, "isend", side_effect=lambda tensor, *a, **kw: record(send, tensor, *a, **kw)):
            handles = coordinator.isend_tensor_dict(output.tensors, all_gather_group=None)
        assert packets == [(tuple(t.shape), t.numel() * t.element_size()) for t in output.tensors.values()]
        for handle in handles:
            handle.wait()
        coordinator._pending_isends.clear()
    else:
        recv = dist.irecv
        with patch.object(dist, "irecv", side_effect=lambda tensor, *a, **kw: record(recv, tensor, *a, **kw)):
            payload, handles, postprocess = coordinator.irecv_tensor_dict(all_gather_group=None)
        assert not postprocess, "The PP transport must retain each rank's shard"
        for handle in handles:
            handle.wait()
        assert payload is not None
        assert packets == [(tuple(t.shape), t.numel() * t.element_size()) for t in payload.values()]
        active = backing[:rows]
        addresses = {key: tensor.data_ptr() for key, tensor in backing.tensors.items()}
        for key, tensor in payload.items():
            assert active[key].shape == tensor.shape
            active[key][:tokens].copy_(tensor[:tokens])  # Same slice contract as MRV2.
        assert not active[AUX_KEY][valid:].count_nonzero()
        torch.testing.assert_close(active[AUX_KEY][:valid], wanted[lane * rows : lane * rows + valid])
        hidden, aux = namespace["forward"](model, None, positions, active)
        torch.testing.assert_close(hidden, wanted + 10)
        assert len(aux) == 2
        for index, snapshot in enumerate(aux):
            torch.testing.assert_close(snapshot, wanted + index * 10)
        assert len(gathers) == 3, "Only the final hidden and two aux gathers are expected"
        assert addresses == {key: tensor.data_ptr() for key, tensor in backing.tensors.items()}
        payload_bytes = sum(t.numel() * t.element_size() for t in payload.values())
        full_bytes = tokens * (HC_STREAMS + 1) * HIDDEN_SIZE * base.element_size()
        if tokens % tp == 0:
            assert payload_bytes * tp == full_bytes
        print(
            json.dumps(
                {
                    "model": "dsv4",
                    "tp_size": tp,
                    "tokens": tokens,
                    "source_rank": lane,
                    "destination_rank": rank,
                    "backend": "hccl" if device.type == "npu" else "gloo",
                    "shard_rows": rows,
                    "main_bytes": payload["hidden_states"].numel() * base.element_size(),
                    "aux_bytes": payload[AUX_KEY].numel() * base.element_size(),
                    "payload_bytes": payload_bytes,
                    "full_payload_bytes": full_bytes,
                    "result": "PASS",
                }
            ),
            flush=True,
        )
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--tp-size", type=int, choices=(2, 8), default=2)
    parser.add_argument("--token-counts", type=int, nargs="+")
    args = parser.parse_args()
    tp = args.tp_size
    counts = args.token_counts or ([8, 17, 1, 8] if tp == 2 else [16, 17, 1, 16])
    if any(count < 1 for count in counts):
        parser.error("token counts must be positive")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 2 * tp
    if args.device == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}" if args.device == "npu" else "cpu")
    repo = next(path for path in Path(__file__).resolve().parents if (path / "vllm_ascend").is_dir())
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    try:
        coordinator_type, namespace = _production(repo, repo.parent / "vllm")
        coordinator, tp_group = _groups(rank, "hccl" if args.device == "npu" else "gloo", coordinator_type, tp)
        capacity = max(counts)
        backing = namespace["IntermediateTensors"](
            {
                "hidden_states": torch.empty((capacity + tp - 1) // tp, HC_STREAMS, HIDDEN_SIZE, device=device),
                AUX_KEY: torch.empty(capacity, HIDDEN_SIZE, device=device),
            }
        )
        for tokens in counts:
            _case(namespace, coordinator, tp_group, rank, tp, tokens, device, backing)
        if rank == 0:
            print(f"PASS PP2TP{tp}: all hidden/aux payloads stay sharded; final outputs match", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
