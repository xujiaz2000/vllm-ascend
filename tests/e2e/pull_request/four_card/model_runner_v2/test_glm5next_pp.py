# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four-NPU MRV2 PP regression with seeded synthetic weights.

Reuse the nine-layer GLM asset; compare TP2/PP2, MTP and DFlash2 with TP2/PP1
target logprobs. CPU UT covers layouts and scheduler gates; this test is needed
for real PP send/recv, sampled-token writeback and NPU cache aliases. Synthetic
weights test numerical consistency, not the accuracy of a trained checkpoint.
"""

import json

from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.four_card.test_glm5_3_flash import LOADER, _write_model
from tests.e2e.pull_request.utils import _check_decode_token, _check_prefill_token

# Use the shared compare_logprobs BF16 tolerance, not an accuracy score.
LOGPROB_ATOL = 0.0689
OUTPUT_TOKENS = 8


def _write_dflash2(destination, target):
    text = target["text_config"]
    config = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "dtype": "bfloat16",
        "hidden_size": text["hidden_size"],
        "intermediate_size": text["hidden_size"],
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 1,
        "num_target_layers": text["num_hidden_layers"],
        "vocab_size": text["vocab_size"],
        "rms_norm_eps": text["rms_norm_eps"],
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "layer_types": ["sliding_attention"],
        "sliding_window": 128,
        "use_sliding_window": True,
        "is_causal": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "selector_rank": 128,
            "selector_top_k": 4,
            # Capture one feature from each PP stage.
            "target_layer_ids": [1, 6],
            "mask_token_id": text["vocab_size"] - 1,
        },
    }
    destination.mkdir()
    (destination / "config.json").write_text(json.dumps(config))


def _generate(model, settings, baseline=None):
    params = SamplingParams(
        temperature=0,
        max_tokens=OUTPUT_TOKENS,
        ignore_eos=True,
        logprobs=20,
        detokenize=False,
    )
    # Force chunked prefill, then exercise decode and mixed request lengths.
    prompts = [{"prompt_token_ids": [10 + i % 1000 for i in range(length)]} for length in (129, 257)]
    with VllmRunner(str(model), **settings) as runner:
        outputs = runner.model.generate(prompts, params, use_tqdm=False)
        metrics = runner.model.get_metrics()
        references = outputs if baseline is None else baseline
        verification_prompts = [
            {"prompt_token_ids": reference.prompt_token_ids + list(reference.outputs[0].token_ids[:token_idx])}
            for reference in references
            for token_idx in range(1, OUTPUT_TOKENS)
        ]
        params.max_tokens = 1
        verified = runner.model.generate(verification_prompts, params, use_tqdm=False)
    return outputs, verified, metrics


def _check_consistency(baseline, outputs, verified_baseline, verified_outputs):
    assert len(outputs) == len(baseline)
    for prompt_idx, (reference, actual) in enumerate(zip(baseline, outputs, strict=True)):
        expected_seq, actual_seq = reference.outputs[0], actual.outputs[0]
        assert actual.finished and len(actual_seq.token_ids) == OUTPUT_TOKENS
        assert expected_seq.logprobs is not None and actual_seq.logprobs is not None
        _check_prefill_token(expected_seq, actual_seq, prompt_idx, LOGPROB_ATOL)
        for token_idx in range(1, OUTPUT_TOKENS):
            # Compare decode directly while both engines see the same history.
            if expected_seq.token_ids[:token_idx] == actual_seq.token_ids[:token_idx]:
                _check_decode_token(expected_seq, actual_seq, token_idx, prompt_idx, 2 * LOGPROB_ATOL)
    # BF16 argmax choices can diverge for synthetic weights. Compare all fixed
    # reference histories using the same prefill path on both engines instead
    # of comparing unrelated contexts or mixing prefill with decode numerics.
    for prompt_idx, (reference, actual) in enumerate(zip(verified_baseline, verified_outputs, strict=True)):
        expected_seq, actual_seq = reference.outputs[0], actual.outputs[0]
        assert actual.finished and len(actual_seq.token_ids) == 1
        assert expected_seq.logprobs is not None and actual_seq.logprobs is not None
        _check_decode_token(expected_seq, actual_seq, 0, prompt_idx, 2 * LOGPROB_ATOL)


def test_glm5next_mrv2_pp_speculative_logprobs(tmp_path, monkeypatch):
    for name, value in {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "400",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    model = tmp_path / "target"
    target = _write_model(model)
    target["text_config"]["num_nextn_predict_layers"] = 1
    (model / "config.json").write_text(json.dumps(target))
    quant_path = model / "quant_model_description.json"
    quant = json.loads(quant_path.read_text())
    # MTP is an MLA/MoE layer past the base stack. Reuse the last MLA layer's
    # synthetic quantization scheme under the draft model's unwrapped prefix.
    source_layer = max(
        index for index, kind in enumerate(target["text_config"]["layer_types"]) if kind == "deepseek_sparse_attention"
    )
    source_prefix = f"model.language_model.layers.{source_layer}."
    draft_prefix = f"model.layers.{target['text_config']['num_hidden_layers']}."
    quant.update(
        {
            draft_prefix + name.removeprefix(source_prefix): value
            for name, value in quant.items()
            if name.startswith(source_prefix)
        }
    )
    quant_path.write_text(json.dumps(quant))
    draft = tmp_path / "draft"
    _write_dflash2(draft, target)
    settings = dict(
        load_format=LOADER,
        worker_extension_cls="tests.e2e.pull_request.four_card.test_glm5_3_flash.FlashWorker",
        dtype="bfloat16",
        quantization="ascend",
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        max_model_len=4096,
        max_num_seqs=2,
        max_num_batched_tokens=256,
        limit_mm_per_prompt={"image": 0, "video": 0},
        block_size=512,
        gpu_memory_utilization=0.8,
        seed=1024,
        disable_log_stats=False,
        additional_config={"enable_cpu_binding": False, "enable_fused_mc2": 0},
    )
    baseline, verified_baseline, _ = _generate(model, {**settings, "pipeline_parallel_size": 1})
    speculative_configs = (
        None,
        {"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": True},
        {"method": "dflash", "model": str(draft), "num_speculative_tokens": 7, "enforce_eager": True},
    )
    for speculative in speculative_configs:
        outputs, verified_outputs, metrics = _generate(
            model,
            {**settings, "pipeline_parallel_size": 2, "speculative_config": speculative},
            baseline,
        )
        _check_consistency(baseline, outputs, verified_baseline, verified_outputs)
        if speculative is not None:
            # Random drafters need not accept tokens; trained-weight acceptance
            # is validated separately. Verify that this is not plain PP fallback.
            assert (
                sum(
                    metric.value
                    for metric in metrics
                    if isinstance(metric, Counter) and metric.name == "vllm:spec_decode_num_drafts"
                )
                > 0
            )
