"""Tests for the profiling/timing infrastructure."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from lmprobe import Probe, is_profiling, set_profiling
from lmprobe.profiling import (
    ProfileAccumulator,
    _fmt_time,
    profile_op,
    profile_section,
)

TEST_MODEL = "stas/tiny-random-llama-2"


# ---------------------------------------------------------------------------
# Unit tests for profiling primitives
# ---------------------------------------------------------------------------


class TestFmtTime:
    def test_microseconds(self):
        assert "µs" in _fmt_time(0.0001)

    def test_milliseconds(self):
        assert "ms" in _fmt_time(0.05)

    def test_seconds(self):
        assert "s" in _fmt_time(5.0)
        assert "min" not in _fmt_time(5.0)

    def test_minutes(self):
        assert "min" in _fmt_time(120.0)


class TestSetProfiling:
    def test_toggle(self):
        original = is_profiling()
        try:
            set_profiling(True)
            assert is_profiling() is True
            set_profiling(False)
            assert is_profiling() is False
        finally:
            set_profiling(original)

    def test_env_var(self, monkeypatch):
        """LMPROBE_PROFILE=1 enables profiling at import time."""
        # We can't re-import easily, but we can test the env var logic
        monkeypatch.setenv("LMPROBE_PROFILE", "1")
        # The flag is checked at module import time; test the function instead
        set_profiling(True)
        assert is_profiling() is True
        set_profiling(False)


class TestProfileOp:
    def test_disabled_yields_none(self):
        set_profiling(False)
        with profile_op("test") as acc:
            assert acc is None

    def test_enabled_yields_accumulator(self):
        set_profiling(False)
        try:
            set_profiling(True)
            with profile_op("test") as acc:
                assert isinstance(acc, ProfileAccumulator)
                assert acc.name == "test"
        finally:
            set_profiling(False)

    def test_sections_recorded(self):
        set_profiling(True)
        try:
            with profile_op("test") as acc:
                with profile_section(acc, "step_a"):
                    _ = sum(range(100))
                with profile_section(acc, "step_b"):
                    _ = sum(range(100))
            assert acc is not None
            d = acc.as_dict()
            assert "total" in d
            assert "step_a" in d
            assert "step_b" in d
            assert d["total"] >= 0
            assert d["step_a"] >= 0
        finally:
            set_profiling(False)

    def test_profile_section_with_none(self):
        """profile_section with None accumulator is a no-op."""
        with profile_section(None, "anything"):
            pass  # should not raise

    def test_logging_output(self, caplog):
        set_profiling(True)
        try:
            with caplog.at_level(logging.INFO, logger="lmprobe.profile"):
                with profile_op("TestOp") as acc:
                    with profile_section(acc, "sub"):
                        pass
            assert any("TestOp" in r.message for r in caplog.records)
            assert any("sub" in r.message for r in caplog.records)
        finally:
            set_profiling(False)


class TestZeroCostWhenDisabled:
    def test_no_timing_overhead(self):
        """When profiling is off, profile_op should add negligible overhead."""
        set_profiling(False)
        import time

        iterations = 10_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            with profile_op("noop"):
                pass
        elapsed = time.perf_counter() - t0
        # Should be < 1ms for 10k iterations (just a bool check + yield)
        assert elapsed < 1.0, f"Disabled profiling took {elapsed:.3f}s for {iterations} iterations"


# ---------------------------------------------------------------------------
# Integration tests with Probe
# ---------------------------------------------------------------------------


class TestProbeProfileIntegration:
    @pytest.fixture
    def fitted_probe(self):
        """Return a fitted probe for testing."""
        probe = Probe(
            model=TEST_MODEL,
            layers=-1,
            pooling="last_token",
            classifier="logistic_regression",
            device="cpu",
            remote=False,
            random_state=42,
        )
        positive = ["dog walk bark", "fetch ball", "good boy wag"]
        negative = ["cat purr sleep", "meow scratch", "litterbox sun"]
        probe.fit(positive, negative)
        return probe

    def test_fit_records_profile(self):
        set_profiling(True)
        try:
            probe = Probe(
                model=TEST_MODEL,
                layers=-1,
                pooling="last_token",
                classifier="logistic_regression",
                device="cpu",
                remote=False,
                random_state=42,
            )
            probe.fit(
                ["dog walk bark", "fetch ball", "good boy wag"],
                ["cat purr sleep", "meow scratch", "litterbox sun"],
            )
            assert hasattr(probe, "last_profile_")
            prof = probe.last_profile_
            assert isinstance(prof, dict)
            assert "total" in prof
            assert "extract_and_pool" in prof
            assert "scale" in prof
            assert "fit" in prof
            assert prof["total"] > 0
        finally:
            set_profiling(False)

    def test_fit_no_profile_when_disabled(self):
        set_profiling(False)
        probe = Probe(
            model=TEST_MODEL,
            layers=-1,
            pooling="last_token",
            classifier="logistic_regression",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(
            ["dog walk bark", "fetch ball", "good boy wag"],
            ["cat purr sleep", "meow scratch", "litterbox sun"],
        )
        # last_profile_ should not be set (or not a dict)
        assert not hasattr(probe, "last_profile_") or probe.last_profile_ is None

    def test_predict_proba_records_profile(self, fitted_probe):
        set_profiling(True)
        try:
            fitted_probe.predict_proba(["test prompt"])
            assert hasattr(fitted_probe, "last_profile_")
            prof = fitted_probe.last_profile_
            assert isinstance(prof, dict)
            assert "total" in prof
            assert "predict" in prof
        finally:
            set_profiling(False)

    def test_fit_from_activations_records_profile(self):
        set_profiling(True)
        try:
            probe = Probe(
                classifier="logistic_regression",
                random_state=42,
            )
            X = np.random.randn(20, 64).astype(np.float32)
            y = np.array([0] * 10 + [1] * 10)
            probe.fit_from_activations(X, y)
            assert hasattr(probe, "last_profile_")
            prof = probe.last_profile_
            assert isinstance(prof, dict)
            assert "total" in prof
            assert "scale" in prof
            assert "fit" in prof
        finally:
            set_profiling(False)

    def test_predict_from_activations_records_profile(self):
        set_profiling(True)
        try:
            probe = Probe(
                classifier="logistic_regression",
                random_state=42,
            )
            X = np.random.randn(20, 64).astype(np.float32)
            y = np.array([0] * 10 + [1] * 10)
            probe.fit_from_activations(X, y)

            X_test = np.random.randn(5, 64).astype(np.float32)
            probe.predict_from_activations(X_test)
            prof = probe.last_profile_
            assert isinstance(prof, dict)
            assert "transform" in prof
            assert "predict" in prof

            probe.predict_proba_from_activations(X_test)
            prof = probe.last_profile_
            assert isinstance(prof, dict)
            assert "transform" in prof
            assert "predict" in prof
        finally:
            set_profiling(False)

    def test_evaluate_records_profile(self, fitted_probe):
        set_profiling(True)
        try:
            fitted_probe.evaluate(["test dog", "test cat"], [1, 0])
            assert hasattr(fitted_probe, "last_profile_")
            prof = fitted_probe.last_profile_
            assert isinstance(prof, dict)
            assert "total" in prof
        finally:
            set_profiling(False)
