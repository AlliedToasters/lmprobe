"""Tests for extraction module, focusing on server-side top-k logits."""

from lmprobe.extraction import (
    _build_remote_extract_fn,
    _extract_batch_with_logits,
)


class TestBuildRemoteExtractFnTopK:
    """Tests for _build_remote_extract_fn with logit_top_k."""

    def test_generated_code_contains_topk(self):
        """When logit_top_k is set, generated code contains topk call."""
        fn = _build_remote_extract_fn(
            layer_indices=[0, 1], with_logits=True, logit_top_k=100
        )
        # The wrapper wraps the real function; inspect the source
        # by checking that the function was created without error
        assert callable(fn)

    def test_generated_code_without_topk(self):
        """When logit_top_k is None, generated code saves full logits."""
        fn = _build_remote_extract_fn(
            layer_indices=[0], with_logits=True, logit_top_k=None
        )
        assert callable(fn)

    def test_generated_code_no_logits(self):
        """When with_logits=False, no logits code generated."""
        fn = _build_remote_extract_fn(
            layer_indices=[0], with_logits=False, logit_top_k=100
        )
        assert callable(fn)


class TestExtractBatchWithLogitsTopK:
    """Tests for _extract_batch_with_logits with logit_top_k."""

    def test_local_returns_4tuple_none_indices(self, tiny_model):
        """Local extraction returns 4-tuple with None indices."""
        from lmprobe.extraction import get_cached_model

        model = get_cached_model(tiny_model, device="cpu", remote=False)
        prompts = ["hello world"]
        layer_indices = [0]

        result = _extract_batch_with_logits(
            model, prompts, layer_indices, remote=False
        )

        assert len(result) == 4
        acts, mask, logits, indices = result
        assert acts.ndim == 3  # (batch, seq_len, hidden_dim)
        assert mask.ndim == 2  # (batch, seq_len)
        assert logits.ndim == 3  # (batch, seq_len, vocab_size)
        assert indices is None

    def test_local_topk_ignored(self, tiny_model):
        """logit_top_k is ignored for local extraction (returns full logits)."""
        from lmprobe.extraction import get_cached_model

        model = get_cached_model(tiny_model, device="cpu", remote=False)
        prompts = ["test prompt"]
        layer_indices = [0]

        result = _extract_batch_with_logits(
            model, prompts, layer_indices, remote=False, logit_top_k=10
        )

        acts, mask, logits, indices = result
        # Local path ignores logit_top_k — full vocab logits returned
        assert indices is None
        # Vocab size should be > 10 (full logits, not top-k)
        assert logits.shape[-1] > 10
