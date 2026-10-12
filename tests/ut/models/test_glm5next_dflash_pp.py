# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.models.interfaces import supports_eagle3

from vllm_ascend.models.glm5next import model as glm
from vllm_ascend.models.glm5next.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration, Glm5NextModel

HIDDEN_SIZE = 8
N_STREAMS = 4


def _model(*, start, layers, mhc=True):
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=HIDDEN_SIZE, mhc=mhc, mhc_num_residual_streams=N_STREAMS)
    model.start_layer = start
    model._active_layers = layers
    model.is_sequence_parallel = False
    model.norm = nn.Identity()
    return model


def _rank(monkeypatch, *, first, last):
    group = SimpleNamespace(world_size=2, is_first_rank=first, is_last_rank=last)
    monkeypatch.setattr(glm, "get_pp_group", lambda: group)
    monkeypatch.setattr("vllm.distributed.parallel_state.get_pp_group", lambda: group)
    monkeypatch.setattr("vllm.distributed.parallel_state.model_parallel_is_initialized", lambda: True)


class _CaptureLayer(nn.Module):
    def __init__(self, *, terminal=False):
        super().__init__()
        self.terminal = terminal
        self.incoming = None
        self.post_calls = 0

    def forward(self, positions, hidden_states, residual, post, comb):
        self.incoming = (post, comb)
        if self.terminal:
            return hidden_states + 1, None, None, None
        shape = (hidden_states.shape[0], N_STREAMS)
        residual = torch.arange(N_STREAMS).view(1, -1, 1).expand(*shape, HIDDEN_SIZE)
        return hidden_states + 1, residual, torch.full((*shape, 1), 2.0), torch.full((*shape, N_STREAMS), 3.0)

    def hc_post(self, hidden_states, residual, post, comb):
        self.post_calls += 1
        return hidden_states.unsqueeze(1) + residual + post + comb.sum(-1).unsqueeze(-1)


def test_mhc_dflash_aux_crosses_pp_without_changing_deferred_target_state(monkeypatch):
    first_layers = [_CaptureLayer(), _CaptureLayer()]
    first = _model(start=0, layers=first_layers)
    _rank(monkeypatch, first=True, last=False)
    inputs = torch.zeros(3, HIDDEN_SIZE)
    args = dict(input_ids=None, positions=torch.arange(3), inputs_embeds=inputs, intermediate_tensors=None)
    baseline = first(**args)
    assert all(layer.post_calls == 0 for layer in first_layers)
    first._set_aux_hidden_state_layers((1, 2, 3))
    payload = first(**args)

    for key, value in baseline.tensors.items():
        torch.testing.assert_close(payload[key], value)
    for index in range(2):
        # Contract only the complete post-FFN mHC state, not the pending FFN output.
        torch.testing.assert_close(payload[f"aux_hidden_states_{index}"], inputs + index + 1 + 15.5)
        assert first_layers[index].post_calls == 1
    assert first_layers[1].incoming[0] is not None
    assert first_layers[1].incoming[1] is not None

    last_layer = _CaptureLayer(terminal=True)
    last = _model(start=2, layers=[last_layer])
    _rank(monkeypatch, first=False, last=True)
    last._set_aux_hidden_state_layers((1, 2, 3))
    hidden, aux = last(input_ids=None, positions=torch.arange(3), intermediate_tensors=payload)
    assert len(aux) == 3
    assert aux[0] is payload["aux_hidden_states_0"]
    assert aux[1] is payload["aux_hidden_states_1"]
    torch.testing.assert_close(aux[2], hidden)
    assert last_layer.post_calls == 0  # The terminal layer has already contracted.
    assert last_layer.incoming == (payload["post"], payload["comb"])


def test_dflash_aux_without_mhc_does_not_add_residual_twice(monkeypatch):
    model = _model(start=0, layers=[_CaptureLayer()], mhc=False)
    _rank(monkeypatch, first=True, last=True)
    model._set_aux_hidden_state_layers((1,))
    hidden, aux = model(
        input_ids=None, positions=torch.arange(2), inputs_embeds=torch.zeros(2, HIDDEN_SIZE), intermediate_tensors=None
    )
    torch.testing.assert_close(aux[0], hidden)
    assert aux[0].shape == (2, HIDDEN_SIZE)


def test_missing_remote_dflash_aux_fails_before_drafting(monkeypatch):
    model = _model(start=2, layers=[])
    _rank(monkeypatch, first=False, last=True)
    model._set_aux_hidden_state_layers((1, 3))
    payload = model.make_empty_intermediate_tensors(2, torch.float32, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="Missing aux_hidden_states_0"):
        model(input_ids=None, positions=torch.arange(2), intermediate_tensors=payload)


@pytest.mark.parametrize("method", [None, "mtp", "dflash"])
@pytest.mark.parametrize("first,last", [(True, False), (False, False), (False, True)])
def test_dflash_last_stage_owns_embedding_for_checkpoint_without_own_embedding(monkeypatch, method, first, last):
    _rank(monkeypatch, first=first, last=last)
    monkeypatch.setattr(glm, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(glm, "make_layers", lambda *_args, **_kwargs: (0, 0, nn.ModuleList()))
    monkeypatch.setattr(glm, "RMSNorm", lambda *_args, **_kwargs: nn.Identity())
    embedding = nn.Embedding(16, HIDDEN_SIZE)
    monkeypatch.setattr(glm, "VocabParallelEmbedding", lambda *_args, **_kwargs: embedding)
    config = SimpleNamespace(
        index_topk=None,
        vocab_size=16,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=1,
        rms_norm_eps=1e-5,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        speculative_config=None if method is None else SimpleNamespace(method=method),
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
    )
    model = Glm5NextModel(vllm_config=vllm_config)
    assert (model.embed_tokens is embedding) == (first or (last and method == "dflash"))


@pytest.mark.parametrize("model_cls", [Glm5NextForCausalLM, Glm5NextForConditionalGeneration])
def test_glm_dflash_target_declares_native_aux_interface(model_cls):
    assert supports_eagle3(model_cls)


def test_quarot_draft_conversion_preserves_features_embedding_and_logits():
    """Guard the tutorial's specified H_rot=H_original@R export convention."""
    generator = torch.Generator().manual_seed(1024)
    hidden_size, num_features = 8, 2
    rotation, _ = torch.linalg.qr(torch.randn(hidden_size, hidden_size, generator=generator, dtype=torch.float64))
    features = torch.randn(3, num_features, hidden_size, generator=generator, dtype=torch.float64)
    fc = torch.randn(hidden_size, num_features, hidden_size, generator=generator, dtype=torch.float64)
    embedding = torch.randn(4, hidden_size, generator=generator, dtype=torch.float64)
    lm_head = torch.randn(4, hidden_size, generator=generator, dtype=torch.float64)
    target_norm = torch.linspace(0.5, 1.5, hidden_size, dtype=torch.float64)

    rotated_features = features @ rotation
    converted_fc = fc @ rotation
    torch.testing.assert_close(
        rotated_features.flatten(1) @ converted_fc.flatten(1).T,
        features.flatten(1) @ fc.flatten(1).T,
    )
    rotated_embedding = embedding @ rotation
    torch.testing.assert_close(rotated_embedding @ rotation.T, embedding)
    folded_rotated_head = (lm_head * target_norm) @ rotation
    recovered_head = (folded_rotated_head @ rotation.T) / target_norm
    torch.testing.assert_close(recovered_head, lm_head)
    draft_hidden = torch.randn(3, hidden_size, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(draft_hidden @ recovered_head.T, draft_hidden @ lm_head.T)
    # An all-ones exported norm cannot undo the original target norm folding.
    assert not torch.allclose(folded_rotated_head @ rotation.T, lm_head)
