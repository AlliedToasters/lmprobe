"""Bit-exactness smoke tests across model families.

Asserts that ``ChunkedLocalBackend`` / ``DiskOffloadBackend`` hidden states
are bit-identical to a dense ``transformers`` reference at every layer.

Background: during a deception-detection audit on llama-3.2-3b we uncovered
four numerical bugs whose signatures (monotonic 1-ULP drift per layer) would
be invisible to a loose ``atol=1e-4`` comparison. Each bug is model-family
specific — the rotary init path differs per ``rope_type``, and the
device-dependent ULP wobble in ``rope_init_fn`` affects every checkpoint
whose rotary is re-initialized (not loaded from safetensors). See issue #284.

**CI-safe tier** (``stas/tiny-random-llama-2`` on CPU): exercises the
``ChunkedLocalBackend`` + ``PreTokenizedPrompts`` + ``apply_final_norm=True``
path on a single prompt. Template for the real-model matrix, minus padding.

**gpu_large tier** (Llama-3.2-3B + Mistral / Qwen2.5 / Gemma-2 /
DeepSeek-V2-Lite): marked ``@pytest.mark.gpu_large`` so it's skipped by
default. Llama-3.2-3B is the known-passing baseline from the audit; the
rest are speculative — each has a family-specific rotary implementation
that may carry the same latent drift we hit on llama.
"""

from __future__ import annotations

import pytest
import torch

from lmprobe.backends import resolve_backend

REAL_MODEL_MATRIX = [
    # (model_id, description, is_baseline)
    # Baseline = confirmed bit-exact during the llama-3.2-3b audit; failure
    # here indicates a regression in the fixes from commits 20ad9e5..ee963d4.
    # The rest are untested — each is a likely regression surface because its
    # rotary path (rope_type / rope_scaling) differs from llama's.
    ("meta-llama/Llama-3.2-3B-Instruct", "baseline: bit-exact on feat/pretokenized-input", True),
    ("mistralai/Mistral-7B-Instruct-v0.2", "standard rope, no scaling", False),
    ("Qwen/Qwen2.5-7B-Instruct", "rope with scaling", False),
    ("google/gemma-2-9b-it", "softcap attention, per-layer-type PE", False),
    ("deepseek-ai/DeepSeek-V2-Lite", "MoE routing, non-trivial kv head structure", False),
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _tokenize_dd_style(
    model_id: str,
    messages_batch: list[list[dict]],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate the deception-detection tokenization path.

    - ``apply_chat_template`` (as plain text) then ``tokenizer(...)`` with
      ``add_special_tokens=False`` avoids double-BOS when the template
      itself emits BOS.
    - ``padding_side="left"`` matches the realistic inference path.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    rendered = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in messages_batch
    ]
    enc = tok(
        rendered,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def _tokenize_plain(
    model_id: str,
    texts: list[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain tokenization fallback for models without a chat template
    (e.g. ``stas/tiny-random-llama-2``)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    enc = tok(texts, padding=True, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def _hf_reference_hidden_states(
    model_id: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    device: str,
) -> list[torch.Tensor]:
    """Run a dense HF forward pass and return ``output.hidden_states``.

    Returns ``N+1`` tensors each of shape ``(B, S, H)``:

    - index ``0``: embedding output
    - index ``k`` for ``k in 1..N``: output of decoder block ``k-1``
    - index ``N``: last entry with ``model.norm`` applied (matches
      ``apply_final_norm=True`` on lmprobe's side)
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    return [hs.detach().cpu() for hs in out.hidden_states]


def _split_per_layer(
    activations: torch.Tensor, n_layers: int,
) -> list[torch.Tensor]:
    """Split ``(B, S, H * L)`` → list of ``L`` tensors each ``(B, S, H)``.

    ``HiddenStateCapture`` writes layer slots in the order of
    ``layer_indices`` (see ``src/lmprobe/accumulators.py:610``), so
    ``layer_indices=list(range(n_layers))`` yields a hidden-dim block per
    layer in ascending order.
    """
    assert activations.dim() == 3, f"expected (B,S,H*L), got {activations.shape}"
    total = activations.shape[-1]
    assert total % n_layers == 0, (
        f"activation width {total} not divisible by n_layers={n_layers}"
    )
    hidden = total // n_layers
    return [
        activations[..., i * hidden : (i + 1) * hidden]
        for i in range(n_layers)
    ]


def _assert_per_layer_bit_exact(
    hf_hidden: list[torch.Tensor],
    lmp_per_layer: list[torch.Tensor],
    *,
    label: str,
) -> None:
    """Compare every layer with exact equality.

    lmprobe layer ``i`` (output of block ``i``) must match
    ``hf_hidden[i+1]``. With ``apply_final_norm=True`` this holds for the
    last layer too.
    """
    assert len(lmp_per_layer) == len(hf_hidden) - 1, (
        f"{label}: lmprobe returned {len(lmp_per_layer)} layers, "
        f"expected {len(hf_hidden) - 1}"
    )
    for i, lmp_i in enumerate(lmp_per_layer):
        ref = hf_hidden[i + 1]
        if not torch.equal(ref, lmp_i):
            diff = (ref.float() - lmp_i.float()).abs()
            raise AssertionError(
                f"{label}: bit-exactness broken at layer {i}: "
                f"mean|diff|={diff.mean():.3e} max|diff|={diff.max():.3e} "
                f"cosine={_cos(ref, lmp_i):.6f}"
            )


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().float()
    bf = b.flatten().float()
    return torch.nn.functional.cosine_similarity(af, bf, dim=0).item()


# ── CI-safe tier: tiny random llama on CPU ───────────────────────────────────


class TestTinyModelBitExactness:
    """Template for the real-model matrix, runnable in CI.

    ``stas/tiny-random-llama-2`` has standard llama rope and random weights;
    it cannot catch family-specific rope bugs but it does exercise:

    - The ``PreTokenizedPrompts`` entry point
    - ``apply_final_norm=True`` last-layer semantics
    - Per-layer slot ordering in ``HiddenStateCapture``

    Padded-batch inputs are **not** exercised here. On CPU with this random
    checkpoint the ChunkedLocalBackend diverges from HF under padding by
    ~1e-3 — likely unrelated to the rotary bugs the gpu_large tier targets,
    and worth tracking as a separate issue (see test_padded_batch_drift
    below, intentionally xfailed).
    """

    def test_chunked_matches_hf_single_prompt(self, tiny_model):
        device = "cpu"
        dtype = torch.float32  # bf16 is not bit-meaningful on CPU
        input_ids, attention_mask = _tokenize_plain(
            tiny_model, ["The quick brown fox jumped over the lazy dog."], device,
        )

        hf_hidden = _hf_reference_hidden_states(
            tiny_model, input_ids, attention_mask, dtype, device,
        )
        n_layers = len(hf_hidden) - 1

        backend = resolve_backend(
            "chunked", tiny_model, device, dtype=dtype, chunk_size=1,
        )
        acts, _ = backend.extract_batch_pretokenized(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_indices=list(range(n_layers)),
            apply_final_norm=True,
        )

        _assert_per_layer_bit_exact(
            hf_hidden,
            _split_per_layer(acts.cpu(), n_layers),
            label=f"{tiny_model}/chunked",
        )

    def test_apply_final_norm_false_diverges_on_last_layer(self, tiny_model):
        """Without ``apply_final_norm``, the last layer differs from HF's
        ``hidden_states[N]`` (HF's last entry has ``model.norm`` applied;
        ours is the raw post-block residual).

        This pins the behaviour the issue calls out: the flag is explicitly
        opt-in, and it is meaningful — a future refactor that silently
        flips the default must break this test.
        """
        device = "cpu"
        dtype = torch.float32
        input_ids, attention_mask = _tokenize_plain(
            tiny_model, ["The quick brown fox."], device,
        )

        hf_hidden = _hf_reference_hidden_states(
            tiny_model, input_ids, attention_mask, dtype, device,
        )
        n_layers = len(hf_hidden) - 1

        backend = resolve_backend(
            "chunked", tiny_model, device, dtype=dtype, chunk_size=1,
        )
        acts, _ = backend.extract_batch_pretokenized(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_indices=list(range(n_layers)),
            apply_final_norm=False,
        )
        lmp = _split_per_layer(acts.cpu(), n_layers)

        # Earlier layers must still match exactly.
        for i in range(n_layers - 1):
            assert torch.equal(hf_hidden[i + 1], lmp[i]), (
                f"layer {i} should match HF without final-norm fix-up"
            )
        # Last layer must differ.
        assert not torch.equal(hf_hidden[-1], lmp[-1]), (
            "last layer unexpectedly matches HF without apply_final_norm; "
            "either the reference already lacks model.norm (check "
            "transformers version) or the flag's default changed"
        )

    @pytest.mark.xfail(
        reason=(
            "ChunkedLocalBackend diverges from HF on padded batches for "
            "stas/tiny-random-llama-2 on CPU (max|diff|≈3e-3 under SDPA, "
            "≈5e-5 under eager). Likely unrelated to the rotary bugs fixed "
            "in 20ad9e5..ee963d4; worth opening a separate issue if it "
            "reproduces on real llama weights."
        ),
        strict=False,
    )
    def test_padded_batch_drift(self, tiny_model):
        """Regression placeholder: tracks the padded-batch drift observed
        on tiny-random-llama. Flip to a passing test once the root cause
        is identified and fixed.
        """
        device = "cpu"
        dtype = torch.float32
        input_ids, attention_mask = _tokenize_plain(
            tiny_model,
            ["The quick brown fox.", "Hello world, this is a test."],
            device,
        )

        hf_hidden = _hf_reference_hidden_states(
            tiny_model, input_ids, attention_mask, dtype, device,
        )
        n_layers = len(hf_hidden) - 1

        backend = resolve_backend(
            "chunked", tiny_model, device, dtype=dtype, chunk_size=1,
        )
        acts, _ = backend.extract_batch_pretokenized(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_indices=list(range(n_layers)),
            apply_final_norm=True,
        )

        _assert_per_layer_bit_exact(
            hf_hidden,
            _split_per_layer(acts.cpu(), n_layers),
            label=f"{tiny_model}/chunked (padded)",
        )


# ── gpu_large tier: real model family matrix ────────────────────────────────


@pytest.mark.gpu_large
@pytest.mark.parametrize(
    "model_id,description,is_baseline", REAL_MODEL_MATRIX,
    ids=[m[0].split("/")[-1] for m in REAL_MODEL_MATRIX],
)
@pytest.mark.parametrize("backend_name", ["chunked", "disk_offload"])
def test_real_model_bit_exactness(
    model_id: str, description: str, is_baseline: bool, backend_name: str,
) -> None:
    """Bit-exactness across (model family) × (backend).

    Skipped unless ``-m gpu_large`` is passed. Run on a ≥24GB GPU.

    Templates the deception-detection audit: dd-style dialogue → chat
    template → ``add_special_tokens=False`` → left padding, then every
    layer is compared to the HF reference with exact equality.

    The llama-3.2-3b row is the known-passing baseline; failure there
    signals a regression in the fixes from commits 20ad9e5..ee963d4. The
    other rows are speculative — each one exercises a model-family-
    specific rotary implementation that may carry the same latent drift
    we hit on llama.
    """
    if not torch.cuda.is_available():
        pytest.skip(f"{backend_name} requires CUDA")

    device = "cuda:0"
    dtype = torch.bfloat16

    messages_batch = [
        [
            {"role": "user", "content": "Hi, who are you?"},
            {"role": "assistant", "content": "I'm an AI assistant."},
        ],
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."},
        ],
    ]
    input_ids, attention_mask = _tokenize_dd_style(
        model_id, messages_batch, device,
    )

    hf_hidden = _hf_reference_hidden_states(
        model_id, input_ids, attention_mask, dtype, device,
    )
    n_layers = len(hf_hidden) - 1

    backend = resolve_backend(backend_name, model_id, device, dtype=dtype)
    acts, _ = backend.extract_batch_pretokenized(
        input_ids=input_ids,
        attention_mask=attention_mask,
        layer_indices=list(range(n_layers)),
        apply_final_norm=True,
    )

    tag = "baseline" if is_baseline else "speculative"
    _assert_per_layer_bit_exact(
        hf_hidden,
        _split_per_layer(acts.cpu(), n_layers),
        label=f"{model_id}/{backend_name} [{tag}] ({description})",
    )
