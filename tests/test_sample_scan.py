"""Tests for SampleScan feature.

Verifies storage I/O, delta capture, scan pipeline, projection,
visualization, and separability queries using synthetic data and
the tiny-random-llama-2 model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import NEGATIVE_PROMPTS, POSITIVE_PROMPTS

TEST_MODEL = "stas/tiny-random-llama-2"

# ---------------------------------------------------------------------------
# Storage tests (no model needed)
# ---------------------------------------------------------------------------


class TestStorage:
    """Roundtrip write/read tests with synthetic data."""

    def test_write_read_metadata_roundtrip(self, tmp_path):
        from lmprobe.scan_storage import ScanMetadata, read_metadata, write_metadata

        meta = ScanMetadata(
            model_id="test-model",
            hidden_dim=64,
            n_layers=4,
            n_components=8,
            n_samples=10,
            creation_date="2026-04-15T00:00:00Z",
            signals=["attn_delta", "mlp_delta"],
        )
        write_metadata(tmp_path, meta)
        loaded = read_metadata(tmp_path)
        assert loaded.model_id == "test-model"
        assert loaded.hidden_dim == 64
        assert loaded.n_layers == 4
        assert loaded.n_components == 8
        assert loaded.signals == ["attn_delta", "mlp_delta"]

    def test_write_read_samples_roundtrip(self, tmp_path):
        from lmprobe.scan_storage import read_samples, write_samples

        write_samples(
            tmp_path,
            sample_ids=[0, 1, 2],
            prompts=["a", "b", "c"],
            labels=[0, 1, 0],
            token_ids_list=[[1, 2], [3, 4, 5], [6]],
            seq_lengths=[2, 3, 1],
        )
        table = read_samples(tmp_path)
        assert table.num_rows == 3
        assert table.column("prompt_text").to_pylist() == ["a", "b", "c"]
        assert table.column("label").to_pylist() == [0, 1, 0]

    def test_write_read_basis_roundtrip(self, tmp_path):
        from lmprobe.scan_storage import read_basis, write_channel

        bases = {
            "attn_delta": np.random.randn(4, 64, 8).astype(np.float16),
            "mlp_delta": np.random.randn(4, 64, 8).astype(np.float16),
        }
        write_channel(
            tmp_path,
            channel_name="0_global",
            bases=bases,
            config={"name": "global", "k": 8, "fit_method": "pca"},
        )
        loaded = read_basis(tmp_path, "0_global")
        assert isinstance(loaded, dict)
        assert set(loaded.keys()) == {"attn_delta", "mlp_delta"}
        np.testing.assert_array_equal(loaded["attn_delta"], bases["attn_delta"])

    def test_read_single_signal_basis(self, tmp_path):
        from lmprobe.scan_storage import read_basis, write_channel

        bases = {
            "attn_delta": np.random.randn(4, 64, 8).astype(np.float16),
        }
        write_channel(tmp_path, "0_global", bases, {"name": "global"})
        loaded = read_basis(tmp_path, "0_global", signal="attn_delta")
        assert isinstance(loaded, np.ndarray)
        np.testing.assert_array_equal(loaded, bases["attn_delta"])

    def test_write_read_channel_config(self, tmp_path):
        from lmprobe.scan_storage import read_channel_config, write_channel

        bases = {"attn_delta": np.zeros((4, 64, 8), dtype=np.float16)}
        config = {"name": "global", "k": 8, "fit_method": "pca"}
        write_channel(tmp_path, "0_global", bases, config)
        loaded = read_channel_config(tmp_path, "0_global")
        assert loaded["name"] == "global"
        assert loaded["k"] == 8

    def test_projections_roundtrip(self, tmp_path):
        from lmprobe.scan_storage import (
            open_projections,
            read_coords,
            write_projections,
        )

        N = 100
        k = 8
        values = np.random.randn(N, 1, k).astype(np.float16)
        sample_ids = np.zeros(N, dtype=np.int32)
        layers = np.repeat(np.arange(4), 25).astype(np.int16)
        token_pos = np.tile(np.arange(25), 4).astype(np.int16)
        signal_idx = np.zeros(N, dtype=np.int8)

        write_projections(
            tmp_path, values, sample_ids, layers, token_pos, signal_idx,
        )

        loaded_values = open_projections(tmp_path)
        assert loaded_values.shape == (N, 1, k)
        np.testing.assert_array_equal(loaded_values, values)

        coords = read_coords(tmp_path)
        assert coords.num_rows == N
        assert "signal" in coords.column_names

    def test_projections_memmap(self, tmp_path):
        from lmprobe.scan_storage import open_projections, write_projections

        N = 50
        values = np.random.randn(N, 1, 4).astype(np.float16)
        write_projections(
            tmp_path, values,
            np.zeros(N, dtype=np.int32),
            np.zeros(N, dtype=np.int16),
            np.zeros(N, dtype=np.int16),
            np.zeros(N, dtype=np.int8),
        )
        mm = open_projections(tmp_path)
        assert isinstance(mm, np.memmap) or isinstance(mm, np.ndarray)
        assert mm.shape == (N, 1, 4)


# ---------------------------------------------------------------------------
# Delta capture tests (requires tiny model)
# ---------------------------------------------------------------------------


class TestDeltaCapture:
    """Test that hooks capture attention and MLP deltas correctly."""

    def test_attn_submodule_found(self, tiny_model):
        from transformers import AutoModelForCausalLM

        from lmprobe.backends import _get_attn_submodule, _get_decoder_layers

        model = AutoModelForCausalLM.from_pretrained(tiny_model)
        layers = _get_decoder_layers(model)
        attn = _get_attn_submodule(layers[0])
        assert attn is not None

    def test_mlp_submodule_found(self, tiny_model):
        from transformers import AutoModelForCausalLM

        from lmprobe.backends import _get_decoder_layers, _get_mlp_submodule

        model = AutoModelForCausalLM.from_pretrained(tiny_model)
        layers = _get_decoder_layers(model)
        mlp = _get_mlp_submodule(layers[0])
        assert mlp is not None

    def test_hook_captures_tensor(self, tiny_model):
        """Verify forward hooks on self_attn and mlp capture tensors."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from lmprobe.backends import (
            _get_attn_submodule,
            _get_decoder_layers,
            _get_mlp_submodule,
        )

        model = AutoModelForCausalLM.from_pretrained(tiny_model)
        tokenizer = AutoTokenizer.from_pretrained(tiny_model)
        model.eval()

        tokens = tokenizer("Hello world", return_tensors="pt")
        layers = _get_decoder_layers(model)
        layer0 = layers[0]

        attn_buf: list[torch.Tensor] = []
        mlp_buf: list[torch.Tensor] = []

        def _attn_hook(_mod, _inp, out, _buf=attn_buf):
            delta = out[0] if isinstance(out, tuple) else out
            _buf.append(delta.detach())

        def _mlp_hook(_mod, _inp, out, _buf=mlp_buf):
            delta = out[0] if isinstance(out, tuple) else out
            _buf.append(delta.detach())

        h1 = _get_attn_submodule(layer0).register_forward_hook(_attn_hook)
        h2 = _get_mlp_submodule(layer0).register_forward_hook(_mlp_hook)

        with torch.no_grad():
            model(**tokens)

        h1.remove()
        h2.remove()

        assert len(attn_buf) == 1
        assert len(mlp_buf) == 1
        assert attn_buf[0].ndim == 3
        assert mlp_buf[0].ndim == 3
        assert attn_buf[0].shape == mlp_buf[0].shape


# ---------------------------------------------------------------------------
# SampleScan.run() integration tests
# ---------------------------------------------------------------------------


class TestScanRun:
    """End-to-end tests for SampleScan.run() with tiny model."""

    @pytest.fixture
    def scan(self, tmp_path):
        """Run a scan and return (scan, scan_dir)."""
        from lmprobe.sample_scan import SampleScan

        prompts = POSITIVE_PROMPTS[:3] + NEGATIVE_PROMPTS[:3]
        labels = [1, 1, 1, 0, 0, 0]

        scan = SampleScan.run(
            prompts=prompts,
            labels=labels,
            model_name=TEST_MODEL,
            scan_dir=tmp_path / "test_scan",
            signals=["attn_delta", "mlp_delta"],
            n_components=4,
            device="cpu",
            batch_size=2,
        )
        return scan, tmp_path / "test_scan"

    def test_creates_directory_structure(self, scan):
        _, scan_dir = scan
        assert (scan_dir / "metadata.json").exists()
        assert (scan_dir / "samples" / "samples.parquet").exists()
        assert (scan_dir / "channels" / "0_global" / "basis_attn_delta.npy").exists()
        assert (scan_dir / "channels" / "0_global" / "basis_mlp_delta.npy").exists()
        assert (scan_dir / "channels" / "0_global" / "config.json").exists()
        assert (scan_dir / "projections" / "values.npy").exists()
        assert (scan_dir / "projections" / "coords.parquet").exists()

    def test_metadata_correct(self, scan):
        scan_obj, scan_dir = scan
        with open(scan_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["model_id"] == TEST_MODEL
        assert meta["n_samples"] == 6
        assert meta["n_components"] == 4
        assert meta["n_layers"] > 0
        assert meta["hidden_dim"] > 0
        assert meta["signals"] == ["attn_delta", "mlp_delta"]

    def test_signals_property(self, scan):
        scan_obj, _ = scan
        assert scan_obj.signals == ["attn_delta", "mlp_delta"]

    def test_bases_per_signal(self, scan):
        scan_obj, _ = scan
        bases = scan_obj.bases
        assert isinstance(bases, dict)
        assert "attn_delta" in bases
        assert "mlp_delta" in bases
        # [n_layers, hidden_dim, k_eff]
        for sig, basis in bases.items():
            assert basis.ndim == 3
            assert basis.shape[0] == scan_obj.n_layers

    def test_projections_shape(self, scan):
        _, scan_dir = scan
        from lmprobe.scan_storage import open_projections

        values = open_projections(scan_dir)
        assert values.ndim == 3
        assert values.shape[1] == 1

    def test_coords_match_projections(self, scan):
        _, scan_dir = scan
        from lmprobe.scan_storage import open_projections, read_coords

        values = open_projections(scan_dir)
        coords = read_coords(scan_dir)
        assert coords.num_rows == values.shape[0]
        assert "signal" in coords.column_names

    def test_load_from_disk(self, scan):
        from lmprobe.sample_scan import SampleScan

        _, scan_dir = scan
        loaded = SampleScan(scan_dir)
        assert loaded.n_samples == 6
        assert loaded.n_layers > 0
        assert loaded.signals == ["attn_delta", "mlp_delta"]


class TestScanWithResidual:
    """Test scan with residual signal."""

    def test_residual_signal(self, tmp_path):
        from lmprobe.sample_scan import SampleScan

        prompts = POSITIVE_PROMPTS[:2] + NEGATIVE_PROMPTS[:2]
        labels = [1, 1, 0, 0]

        scan = SampleScan.run(
            prompts=prompts,
            labels=labels,
            model_name=TEST_MODEL,
            scan_dir=tmp_path / "res_scan",
            signals=["residual"],
            n_components=4,
            device="cpu",
            batch_size=2,
        )
        assert scan.signals == ["residual"]
        assert "residual" in scan.bases


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------


class TestProjectPrompt:
    """Test projecting a new prompt onto an existing scan basis."""

    @pytest.fixture
    def scan(self, tmp_path):
        from lmprobe.sample_scan import SampleScan

        prompts = POSITIVE_PROMPTS[:3] + NEGATIVE_PROMPTS[:3]
        labels = [1, 1, 1, 0, 0, 0]

        return SampleScan.run(
            prompts=prompts,
            labels=labels,
            model_name=TEST_MODEL,
            scan_dir=tmp_path / "proj_scan",
            n_components=4,
            device="cpu",
            batch_size=2,
        )

    def test_project_shape(self, scan):
        projections, tokens, log_probs = scan.project_prompt("Hello world")
        # [seq_len, n_layers, n_signals, k]
        assert projections.ndim == 4
        assert projections.shape[1] == scan.n_layers
        assert projections.shape[2] == len(scan.signals)
        assert len(tokens) == projections.shape[0]

    def test_project_single_signal(self, scan):
        projections, tokens, _ = scan.project_prompt("Hello world", signal="attn_delta")
        assert projections.shape[2] == 1  # single signal

    def test_project_has_log_probs(self, scan):
        _, _, log_probs = scan.project_prompt("Hello world")
        assert log_probs is not None
        assert log_probs.ndim == 1

    def test_project_differs_between_prompts(self, scan):
        proj1, _, _ = scan.project_prompt("The dog barked loudly at the mailman")
        proj2, _, _ = scan.project_prompt("Quantum mechanics is fundamentally strange")
        min_len = min(proj1.shape[0], proj2.shape[0])
        assert not np.allclose(
            proj1[:min_len, -1, :, :], proj2[:min_len, -1, :, :],
        )


# ---------------------------------------------------------------------------
# Grouped batch projection
# ---------------------------------------------------------------------------


class TestBatchProjectGrouped:
    """Verify fused multi-group projection — spec 001."""

    @pytest.fixture
    def scan(self, tmp_path):
        from lmprobe.sample_scan import SampleScan

        prompts = POSITIVE_PROMPTS[:3] + NEGATIVE_PROMPTS[:3]
        labels = [1, 1, 1, 0, 0, 0]

        return SampleScan.run(
            prompts=prompts,
            labels=labels,
            model_name=TEST_MODEL,
            scan_dir=tmp_path / "grouped_scan",
            n_components=4,
            device="cpu",
            batch_size=2,
        )

    def test_single_group_matches_batch_project(self, scan):
        prompts = ["The dog ran fast", "The cat slept quietly"]

        proj_ref, coords_ref, tokens_ref, lens_ref = scan.batch_project(prompts)
        out = scan.batch_project_grouped({"only": prompts})

        assert list(out.keys()) == ["only"]
        proj, coords, tokens, lens = out["only"]

        np.testing.assert_array_equal(proj, proj_ref)
        assert tokens == tokens_ref
        assert lens == lens_ref
        for key in coords_ref:
            assert list(coords[key]) == list(coords_ref[key])

    def test_multi_group_matches_per_group(self, scan):
        g_a = ["The dog ran fast"]
        g_b = ["The cat slept", "A bird flew"]

        proj_a, coords_a, tokens_a, lens_a = scan.batch_project(g_a)
        proj_b, coords_b, tokens_b, lens_b = scan.batch_project(g_b)

        out = scan.batch_project_grouped({"a": g_a, "b": g_b})

        np.testing.assert_array_equal(out["a"][0], proj_a)
        np.testing.assert_array_equal(out["b"][0], proj_b)
        assert out["a"][2] == tokens_a
        assert out["b"][2] == tokens_b
        assert out["a"][3] == lens_a
        assert out["b"][3] == lens_b
        for key in coords_a:
            assert list(out["a"][1][key]) == list(coords_a[key])
        for key in coords_b:
            assert list(out["b"][1][key]) == list(coords_b[key])

    def test_sample_id_rebased(self, scan):
        out = scan.batch_project_grouped({
            "a": ["The dog ran"],
            "b": ["The cat slept", "A bird flew"],
        })
        assert set(out["a"][1]["sample_id"]) == {0}
        assert set(out["b"][1]["sample_id"]) == {0, 1}

    def test_preserves_key_order(self, scan):
        keys = ["z", "a", "m", "b"]
        groups = {k: [f"prompt for {k}"] for k in keys}
        out = scan.batch_project_grouped(groups)
        assert list(out.keys()) == keys

    def test_one_backend_call_for_n_groups(self, scan):
        call_count = {"n": 0}
        real_scan_forward = scan._get_backend().scan_forward

        def counting_scan_forward(*args, **kwargs):
            call_count["n"] += 1
            return real_scan_forward(*args, **kwargs)

        scan._get_backend().scan_forward = counting_scan_forward
        try:
            scan.batch_project_grouped({
                "a": ["The dog ran"],
                "b": ["The cat slept"],
                "c": ["A bird flew"],
            })
        finally:
            scan._get_backend().scan_forward = real_scan_forward

        assert call_count["n"] == 1

    def test_batch_project_warns_on_third_call(self, scan):
        import warnings

        with warnings.catch_warnings(record=True) as w1:
            warnings.simplefilter("always")
            scan.batch_project(["p1"])
        assert not any("batch_project_grouped" in str(x.message) for x in w1)

        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            scan.batch_project(["p2"])
        assert not any("batch_project_grouped" in str(x.message) for x in w2)

        with warnings.catch_warnings(record=True) as w3:
            warnings.simplefilter("always")
            scan.batch_project(["p3"])
        matching = [x for x in w3 if "batch_project_grouped" in str(x.message)]
        assert len(matching) == 1
        assert issubclass(matching[0].category, UserWarning)

    def test_batch_project_warning_fires_once(self, scan):
        import warnings

        for _ in range(3):
            scan.batch_project(["p"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scan.batch_project(["p"])
            scan.batch_project(["p"])
        assert not any("batch_project_grouped" in str(x.message) for x in w)

    def test_grouped_does_not_trigger_warning(self, scan):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for _ in range(5):
                scan.batch_project_grouped({"k": ["p"]})
        assert not any("batch_project_grouped" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# Spec 002: stream-project parity
# ---------------------------------------------------------------------------


class TestStreamProjectParity:
    """Projecting through an external basis via the stream-project path
    (delta @ basis on device, per-microbatch) must agree with projecting
    through the legacy captures → stack → PCA.transform path, up to
    fp16 / GPU-vs-CPU matmul tolerance."""

    @pytest.fixture
    def scan(self, tmp_path):
        from lmprobe.sample_scan import SampleScan

        prompts = POSITIVE_PROMPTS[:3] + NEGATIVE_PROMPTS[:3]
        labels = [1, 1, 1, 0, 0, 0]

        return SampleScan.run(
            prompts=prompts,
            labels=labels,
            model_name=TEST_MODEL,
            scan_dir=tmp_path / "parity_scan",
            signals=["attn_delta", "mlp_delta"],
            n_components=4,
            device="cpu",
            batch_size=2,
        )

    def test_batch_project_matches_stored_projections(self, scan):
        """Re-projecting the fit corpus through its own basis reproduces
        the projections computed during the PCA-fit pass (legacy path).

        The fit path does ``pca.transform(flat)`` on CPU in fp32. The
        stream-project path does ``(delta.reshape(-1,H).float() @ basis)``
        on the same device. The basis is identical (``pca.components_.T``),
        so outputs should match within fp16 tolerance."""
        prompts = POSITIVE_PROMPTS[:3] + NEGATIVE_PROMPTS[:3]

        proj_stream, coords_stream, _, seq_lens = scan.batch_project(prompts)

        # Assemble per-sample dense tensor from stream output:
        # [seq_len, n_layers, n_signals, k]
        n_layers = scan.n_layers
        n_signals = len(scan.signals)
        k = scan.n_components

        sample_ids = np.asarray(coords_stream["sample_id"])
        layers = np.asarray(coords_stream["layer"])
        tokens = np.asarray(coords_stream["token_pos"])
        sigs = np.asarray(coords_stream["signal"])

        for i in range(len(prompts)):
            ref = scan.get_projections(i)  # from stored legacy-fit projections
            seq_len = ref.shape[0]

            mask = sample_ids == i
            stream_dense = np.zeros(
                (seq_len, n_layers, n_signals, k), dtype=np.float32,
            )
            rows = np.where(mask)[0]
            for r in rows:
                t = int(tokens[r])
                if t >= seq_len:
                    continue  # padding token beyond real seq_len
                L = int(layers[r])
                s = int(sigs[r])
                stream_dense[t, L, s, :] = proj_stream[r, 0, :].astype(
                    np.float32,
                )

            np.testing.assert_allclose(
                stream_dense, ref, atol=1e-2, rtol=1e-2,
                err_msg=f"sample {i}: stream-project diverges from PCA-fit",
            )

    def test_stream_project_dominates_when_external_bases_given(self, scan):
        """Instrument the backend: under stream-project, the
        ``per_layer_captures`` / ``per_signal_captures`` bucket should
        never accumulate cross-batch CPU tensors for signals that have a
        basis. We verify indirectly: ``batch_project`` returns identical
        shapes and non-zero projections."""
        prompts = ["Hello world", "Another test prompt"]
        proj, coords, _, seq_lens = scan.batch_project(prompts)

        # Output shape: N_samples * max_seq_len (padded) * n_layers * n_signals
        # rows of [k]-vectors. Rows for padded positions are projected but
        # will get filtered by callers using seq_lens.
        n_signals = len(scan.signals)
        n_layers = scan.n_layers
        max_seq = max(seq_lens)
        expected_rows = len(prompts) * max_seq * n_layers * n_signals
        assert proj.shape[0] == expected_rows
        assert proj.shape[2] == scan.n_components
        # At least some projections are non-zero (sanity: stream-project
        # actually produced output rather than leaving zeros).
        assert np.abs(proj).sum() > 0


# ---------------------------------------------------------------------------
# Get projections from stored corpus data
# ---------------------------------------------------------------------------


class TestGetProjections:
    """Test retrieval of stored projections for corpus samples."""

    @pytest.fixture
    def scan(self, tmp_path):
        from lmprobe.sample_scan import SampleScan

        prompts = POSITIVE_PROMPTS[:3] + NEGATIVE_PROMPTS[:3]
        labels = [1, 1, 1, 0, 0, 0]

        return SampleScan.run(
            prompts=prompts,
            labels=labels,
            model_name=TEST_MODEL,
            scan_dir=tmp_path / "get_proj_scan",
            n_components=4,
            device="cpu",
            batch_size=2,
        )

    def test_get_projections_shape(self, scan):
        proj = scan.get_projections(sample_id=0)
        # [seq_len, n_layers, n_signals, k]
        assert proj.ndim == 4
        assert proj.shape[1] == scan.n_layers
        assert proj.shape[2] == len(scan.signals)

    def test_get_projections_single_signal(self, scan):
        proj = scan.get_projections(sample_id=0, signal="attn_delta")
        assert proj.shape[2] == 1


_has_matplotlib = True
try:
    import matplotlib  # noqa: F401
except ImportError:
    _has_matplotlib = False

# ---------------------------------------------------------------------------
# Plot tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_matplotlib, reason="matplotlib not installed")
class TestPlot:
    """Verify the hero figure renders without error."""

    @pytest.fixture
    def scan(self, tmp_path):
        from lmprobe.sample_scan import SampleScan

        prompts = POSITIVE_PROMPTS[:3] + NEGATIVE_PROMPTS[:3]
        labels = [1, 1, 1, 0, 0, 0]

        return SampleScan.run(
            prompts=prompts,
            labels=labels,
            model_name=TEST_MODEL,
            scan_dir=tmp_path / "plot_scan",
            n_components=4,
            device="cpu",
            batch_size=2,
        )

    def test_plot_returns_figure(self, scan):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.figure

        fig = scan.plot("Hello world")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_single_signal(self, scan):
        import matplotlib

        matplotlib.use("Agg")
        fig = scan.plot("Hello world", signal="attn_delta")
        assert fig is not None

    def test_plot_all_signals(self, scan):
        import matplotlib

        matplotlib.use("Agg")
        fig = scan.plot("Hello world")  # default: all signals
        assert fig is not None

    def test_plot_no_surprise(self, scan):
        import matplotlib

        matplotlib.use("Agg")
        fig = scan.plot("Hello world", show_surprise=False)
        assert fig is not None


# ---------------------------------------------------------------------------
# Separability tests
# ---------------------------------------------------------------------------


class TestSeparability:
    """Test separability_map query API."""

    @pytest.fixture
    def scan(self, tmp_path):
        from lmprobe.sample_scan import SampleScan

        prompts = POSITIVE_PROMPTS[:3] + NEGATIVE_PROMPTS[:3]
        labels = [1, 1, 1, 0, 0, 0]

        return SampleScan.run(
            prompts=prompts,
            labels=labels,
            model_name=TEST_MODEL,
            scan_dir=tmp_path / "sep_scan",
            n_components=4,
            device="cpu",
            batch_size=2,
        )

    def test_separability_map_shape(self, scan):
        result = scan.separability_map()
        assert result.shape == (scan.n_layers, len(scan.signals))

    def test_separability_values_bounded(self, scan):
        result = scan.separability_map()
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_separability_single_signal(self, scan):
        result = scan.separability_map(signal="mlp_delta")
        assert result.shape == (scan.n_layers, 1)
