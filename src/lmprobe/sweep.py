"""One-pass sweep primitive with pluggable accumulators.

A ``sweep`` iterates a model forward once over a corpus of prompts, emitting
per-(layer, signal) deltas to registered :class:`Accumulator` instances. The
driver is backend-agnostic — it sees a :class:`LayerLoader` that knows how to
materialize layers (CPU streaming vs. disk streaming) and a bag of
accumulators that declare which signals and layers they care about.

Two invariants drive the design:

1. **One projection per (layer, signal) per microbatch.** If any accumulator
   subscribes to a signal with a projection-backed basis, the driver
   stream-projects on GPU exactly once and hands the resulting ``[B, S, k]``
   fp16 ndarray to every projection-wanting subscriber. The raw
   ``[B, S, H]`` delta never touches CPU.
2. **Raw and projection subscribers don't mix on the same signal.** PCAFit
   needs the CPU delta; PerTokenProjection needs the projection. Both at
   once doubles the bandwidth for no reason, and PCAFit's basis isn't
   known at capture time anyway. The driver enforces this — use two
   sweeps for fit-then-project (see ``SampleScan.run``).

See spec 003 in the repo for the motivation. Reducers from that spec
remain first-class citizens — they're just accumulators with a
per-sample reduction shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from .activation_types import PreTokenizedPrompts


# Valid signal names the driver understands natively. Accumulators may
# declare additional custom signals, but built-in capture only handles
# these.
SIGNAL_RESIDUAL = "residual"
SIGNAL_ATTN_DELTA = "attn_delta"
SIGNAL_MLP_DELTA = "mlp_delta"
SIGNAL_ROUTER_LOGITS = "router_logits"
SIGNAL_LOGITS = "logits"  # end-of-sweep, not a per-layer signal

VALID_SIGNALS: frozenset[str] = frozenset({
    SIGNAL_RESIDUAL,
    SIGNAL_ATTN_DELTA,
    SIGNAL_MLP_DELTA,
    SIGNAL_ROUTER_LOGITS,
    SIGNAL_LOGITS,
})

# Signals captured via module forward hooks (driver registers a hook).
_HOOKED_SIGNALS: frozenset[str] = frozenset({
    SIGNAL_ATTN_DELTA,
    SIGNAL_MLP_DELTA,
    SIGNAL_ROUTER_LOGITS,
})

# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass
class EmbedState:
    """What :meth:`LayerLoader.prepare` hands back to the driver.

    Carries all per-sweep tokenization + embedding + rotary artifacts the
    per-layer loop needs. The residual buffer is opaque to the driver —
    access it via :meth:`LayerLoader.read_hs` / :meth:`LayerLoader.write_hs`.
    """

    input_ids: torch.Tensor                       # [N, S_max] long, CPU
    attention_mask: torch.Tensor                  # [N, S_max] long, CPU
    batches: list[tuple[int, int]]                # microbatch (start, end) pairs
    pos_ids_per_batch: list[torch.Tensor]          # len = n_batches, each [B, S]
    cache_positions_per_batch: list[torch.Tensor]  # len = n_batches, each [S]
    position_embeddings: Any                       # tuple[Tensor] | dict[str, tuple] | None
    layer_types: list[str] | None                  # Gemma3-style, if applicable
    seq_lengths: list[int]                         # per-sample real token counts
    token_ids_per_sample: list[list[int]]          # unpadded
    hidden_dim: int
    residual_buffer: Any                           # loader-private; passed back to read/write


@dataclass
class SweepContext:
    """What each accumulator receives at ``init`` time.

    ``k_per_sig`` is populated when ``external_bases`` are supplied — lets
    accumulators size their outputs correctly even when ``k_eff < n_components``
    varies per (layer, signal).
    """

    n_samples: int
    num_layers: int
    signals: list[str]            # driver-wide signal list (union of subscriptions)
    signal_dims: dict[str, int]   # sig -> hidden_dim or n_experts; filled before init when known
    hidden_dim: int
    dtype: torch.dtype
    device: str
    seq_lengths: list[int]        # per-sample real token counts, len == n_samples
    s_max: int                    # tokenizer-padded sequence length
    # Per-(sig, layer) rank of the supplied basis, or None if no external basis
    # for that signal. Layout: k_per_sig[sig][layer_idx] = int.
    k_per_sig: dict[str, list[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class LayerLoader(Protocol):
    """Backend strategy for layer weight lifecycle + embedding.

    Two concrete implementations will live in ``backends.py``:
    ``ChunkedLayerLoader`` (full model on CPU, chunks streamed to GPU,
    memmap-backed residuals) and ``DiskOffloadLayerLoader`` (per-layer
    safetensors materialization, list-backed residuals).

    The loader owns the residual buffer: the driver reads/writes via
    :meth:`read_hs` / :meth:`write_hs` and never touches the backing
    storage.
    """

    # --- required attributes -------------------------------------------------
    tokenizer: PreTrainedTokenizerBase
    num_layers: int
    hidden_dim: int
    dtype: torch.dtype
    device: str
    layer_types: list[str] | None                  # Gemma3 per-layer-type PE
    router_module_template: str | None             # MoE, e.g. "model.layers.{layer}.mlp"

    # --- lifecycle -----------------------------------------------------------
    def prepare(
        self,
        prompts: list[str] | PreTokenizedPrompts,
        batch_size: int,
    ) -> AbstractContextManager[EmbedState]:
        """Tokenize, embed, compute rotary PE, allocate residual buffer.

        Returns a context manager so the residual-buffer tempdir (memmap
        path for ChunkedLayerLoader) is deterministically cleaned up on
        sweep exit — normal or exceptional. Implementations typically
        decorate the body with ``@contextmanager``.
        """
        ...

    def iter_layer_groups(self) -> Iterable[list[int]]:
        """Yield layer index groups in forward order.

        Chunked yields ``[0..cs-1], [cs..2cs-1], ...`` where ``cs`` is the
        VRAM-fit chunk size. DiskOffload yields ``[0], [1], ...`` — each
        layer is its own group.
        """
        ...

    def layer_group(
        self, indices: list[int],
    ) -> AbstractContextManager[list[Any]]:
        """Context-manage layer weights for ``indices`` being on-device.

        Implementations should be decorated ``@contextmanager`` and yield
        the corresponding ``nn.Module`` list. On exit — normal or
        exceptional — weights are freed back to CPU / unmaterialized.
        Kept as a context manager so the sweep can't forget to tear down
        on exception.
        """
        ...

    def read_hs(
        self,
        state: EmbedState,
        start: int,
        end: int,
        device: str,
    ) -> torch.Tensor:
        """Read ``hs[start:end]`` from the residual buffer onto ``device``."""
        ...

    def write_hs(
        self,
        state: EmbedState,
        start: int,
        end: int,
        hs: torch.Tensor,
    ) -> None:
        """Write ``hs`` back into the residual buffer at ``[start:end]``."""
        ...

    def build_layer_kwargs(
        self,
        state: EmbedState,
        batch_idx: int,
        layer_idx: int,
        causal_mask_dev: torch.Tensor,
        device: str,
    ) -> dict[str, Any]:
        """Assemble per-layer forward kwargs (attention_mask, position_ids,
        cache_position, position_embeddings) for the given batch + layer."""
        ...

    def apply_lm_head(
        self,
        final_hs: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Apply final_norm + lm_head to the final residual state.

        Called by the driver at end-of-sweep if any accumulator subscribes
        to the ``"logits"`` signal. ``final_hs`` is on ``device``; return
        ``[N, S, V]`` on device (driver handles ``.cpu()`` dispatch).
        """
        ...


@runtime_checkable
class Accumulator(Protocol):
    """Consumes per-microbatch signals and produces an output at sweep end.

    Subclasses declare:

    - ``signals``: which signal names trigger ``update()``. Non-matching
      signals are skipped by the driver — zero overhead.
    - ``layers``: which layer indices trigger ``update()``. ``None`` =
      all layers.
    - ``wants_raw``: ``True`` if the accumulator needs the raw delta on
      CPU (e.g. PCAFit). ``False`` if it consumes the stream-projected
      output (e.g. LastTokenReducer, PerTokenProjection). A signal with
      BOTH raw and projection subscribers is rejected at resolve time —
      use two sweeps (fit, then project).

    Attribute convention — set these on the **instance** from ``__init__``,
    not as class-level defaults. The resolver reads them per-accumulator
    once at sweep start, so either works today; pinning to instance form
    avoids subclass authors accidentally sharing class-level state (e.g.
    a mutable ``frozenset`` default that's fine, a ``None`` override that
    isn't). Built-in accumulators follow this convention.
    """

    # Class- or instance-level attributes. Not properties — the driver
    # reads them per accumulator at sweep start, not per update.
    signals: frozenset[str]
    layers: frozenset[int] | None
    wants_raw: bool

    def init(self, ctx: SweepContext) -> None:
        """Allocate accumulator state. Called once at sweep start."""
        ...

    def update(
        self,
        data: np.ndarray | torch.Tensor,
        sig: str,
        layer_idx: int,
        sample_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> None:
        """Fold one microbatch's contribution.

        ``data`` is either:
          - a CPU ``torch.Tensor`` (wants_raw=True) — shape ``[B, S, H]``
            for activation deltas, ``[B, S, E]`` for ``router_logits``,
            ``[B, S, V]`` for ``logits``.
          - a CPU ``np.ndarray`` (wants_raw=False) — shape ``[B, S, k]``
            fp16, the result of stream-projecting the delta through the
            external basis.

        Called once per ``(layer_idx, sig)`` subscription per microbatch.
        """
        ...

    def finalize(self) -> Any:
        """Return the accumulator's output. Called once at sweep end."""
        ...

    # Optional, duck-typed hook. Not part of this Protocol because
    # ``runtime_checkable`` bails on Protocols with optional methods.
    # See :class:`LifecycleAccumulator` for a non-runtime-checkable
    # sub-protocol that mypy can type-check against.


class LifecycleAccumulator(Protocol):
    """Non-runtime-checkable sub-protocol for accumulators that opt into
    layer-group lifecycle hooks.

    Implement this when your accumulator needs to act after a layer group
    completes (e.g. :class:`PCAFit` fits-and-frees incrementally per group
    to keep captures from accumulating across the whole sweep, which would
    regress the per-chunk memory ceiling spec 004 holds).

    The driver finds the hook via ``getattr(acc, "on_layer_group_complete",
    None)`` so a full :class:`Accumulator` needn't inherit from this —
    declaring the attribute is enough. Inheriting just lets mypy catch
    typos on the method name.
    """

    signals: frozenset[str]
    layers: frozenset[int] | None
    wants_raw: bool

    def init(self, ctx: SweepContext) -> None: ...
    def update(
        self,
        data: np.ndarray | torch.Tensor,
        sig: str,
        layer_idx: int,
        sample_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> None: ...
    def finalize(self) -> Any: ...
    def on_layer_group_complete(self, layer_indices: list[int]) -> None:
        """Called after every microbatch has visited ``layer_indices`` and
        right before the group's weights are freed."""
        ...


# ---------------------------------------------------------------------------
# Subscription resolver
# ---------------------------------------------------------------------------


@dataclass
class _SignalPlan:
    """Per-(layer, signal) dispatch plan computed once at sweep start."""

    needs_raw: bool
    needs_projection: bool
    raw_subs: list[Accumulator]
    proj_subs: list[Accumulator]


def _collect_signals(
    accumulators: Mapping[str, Accumulator],
) -> list[str]:
    """Union of signal names across all accumulators, in insertion order
    of first declaration. Deduplicated."""
    seen: dict[str, None] = {}
    for acc in accumulators.values():
        for sig in acc.signals:
            if sig not in VALID_SIGNALS:
                raise ValueError(
                    f"accumulator declared unknown signal {sig!r}; "
                    f"valid: {sorted(VALID_SIGNALS)}"
                )
            seen.setdefault(sig, None)
    return list(seen)


def _resolve_plans(
    accumulators: Mapping[str, Accumulator],
    num_layers: int,
    external_bases: Mapping[str, np.ndarray] | None,
) -> dict[tuple[int, str], _SignalPlan]:
    """Compute per-(layer, signal) dispatch plans, enforcing:

    - raw + projection on same signal → error (two-sweep rule)
    - projection without external_bases[sig] → error

    ``SIGNAL_LOGITS`` is end-of-sweep, not per-layer — it is dispatched
    once after the final layer completes (see the loader's ``_drive_sweep``)
    and does not appear in ``plans`` at all. Excluding it from the
    per-layer loop avoids generating spurious plan entries that would
    double-fire if a future refactor ever routed logits through
    :func:`_dispatch_plan`.
    """
    external_bases = external_bases or {}
    plans: dict[tuple[int, str], _SignalPlan] = {}
    per_layer_signals = VALID_SIGNALS - {SIGNAL_LOGITS}

    for layer_idx in range(num_layers):
        for sig in per_layer_signals:
            subs = [
                a for a in accumulators.values()
                if sig in a.signals
                and (a.layers is None or layer_idx in a.layers)
            ]
            if not subs:
                continue
            raw_subs = [a for a in subs if a.wants_raw]
            proj_subs = [a for a in subs if not a.wants_raw]

            if raw_subs and proj_subs:
                names = [
                    n for n, a in accumulators.items() if a in subs
                ]
                raise ValueError(
                    f"sweep: signal {sig!r} at layer {layer_idx} has both "
                    f"raw ({[type(a).__name__ for a in raw_subs]}) and "
                    f"projection ({[type(a).__name__ for a in proj_subs]}) "
                    f"subscribers. Accumulators: {names}. These can't share "
                    f"a sweep — project-wanting accumulators need a basis "
                    f"that raw-wanting accumulators haven't fit yet. Use "
                    f"two sweeps: one with PCAFit (raw), then one with "
                    f"the projection accumulators. See SampleScan.run for "
                    f"the reference pattern."
                )
            if proj_subs and sig not in external_bases and sig != SIGNAL_LOGITS:
                # Logits is a special case — lm_head gives us the "projection"
                # directly, no basis matmul needed.
                raise ValueError(
                    f"sweep: signal {sig!r} has projection subscribers "
                    f"({[type(a).__name__ for a in proj_subs]}) but no "
                    f"external_bases[{sig!r}] was supplied. Either supply "
                    f"a basis for this signal or use a raw-wanting "
                    f"accumulator (e.g. PCAFit)."
                )
            plans[(layer_idx, sig)] = _SignalPlan(
                needs_raw=bool(raw_subs),
                needs_projection=bool(proj_subs),
                raw_subs=raw_subs,
                proj_subs=proj_subs,
            )
    return plans


# ---------------------------------------------------------------------------
# Built-in helpers
# ---------------------------------------------------------------------------


def _stream_project(
    delta: torch.Tensor,
    basis_gpu: torch.Tensor,
) -> np.ndarray:
    """Project ``[B, S, H]`` GPU tensor through ``[H, k]`` fp32 GPU basis.

    Returns fp16 CPU ``np.ndarray`` of shape ``[B, S, k]``. The upcast to
    fp32 before matmul matches the legacy CPU fp32 ``pca.transform`` path
    — preserves numerical parity with scans fit before spec 002.
    """
    B, S, H = delta.shape
    flat = delta.reshape(-1, H).to(torch.float32)
    proj = (flat @ basis_gpu).reshape(B, S, -1).to(torch.float16)
    return proj.cpu().numpy()


def _dispatch_plan(
    plan: _SignalPlan,
    delta_gpu: torch.Tensor,
    sig: str,
    layer_idx: int,
    sample_ids: np.ndarray,
    attention_mask_np: np.ndarray,
    basis_gpu: torch.Tensor | None,
) -> None:
    """Dispatch one captured signal to its subscribers per the resolved plan."""
    if plan.needs_raw:
        data = delta_gpu.detach().cpu()
        for sub in plan.raw_subs:
            sub.update(data, sig, layer_idx, sample_ids, attention_mask_np)
        del data
    if plan.needs_projection:
        assert basis_gpu is not None, "resolver guarantees basis for proj"
        proj = _stream_project(delta_gpu, basis_gpu)
        for sub in plan.proj_subs:
            sub.update(proj, sig, layer_idx, sample_ids, attention_mask_np)


def _notify_group_complete(
    accumulators: Mapping[str, Accumulator],
    layer_indices: list[int],
) -> None:
    """Duck-typed broadcast of ``on_layer_group_complete`` to accumulators
    that opt in. PCAFit uses this to fit-and-free incrementally so captures
    don't accumulate past a single layer group's worth of raw deltas."""
    for acc in accumulators.values():
        hook = getattr(acc, "on_layer_group_complete", None)
        if hook is not None:
            hook(layer_indices)


# ---------------------------------------------------------------------------
# Pre-flight memory estimate (issue #275)
# ---------------------------------------------------------------------------


def _fmt_gib(nbytes: int) -> str:
    return f"{nbytes / (1024**3):.1f} GiB"


def _preflight_memory_check(
    loader: LayerLoader,
    ctx: SweepContext,
    accumulators: Mapping[str, Accumulator],
    batch_size: int,
) -> None:
    """Estimate peak RAM and warn if it approaches system-available RAM.

    This is a *best-effort* heuristic, not a hard raise. The model is
    already loaded (its bytes are in RSS), so we only count what the
    sweep is about to *add*:

    1. Per-microbatch capture buffers held on CPU for each active signal
       across the current layer group:
       ``chunk_size × n_signals_hooked × batch_size × s_max × hidden × dtype_bytes``
    2. Accumulator pre-allocations, summed via each accumulator's optional
       :meth:`estimate_bytes(ctx, batch_size)` (duck-typed; defaults to 0).

    If (new allocations) ≥ 90% of ``psutil.virtual_memory().available``,
    emit a :class:`UserWarning` with a breakdown and a halving suggestion.

    ``psutil`` is optional — if it isn't importable, the check is a no-op.
    """
    try:
        import psutil
    except ImportError:
        return

    dtype_bytes = torch.tensor([], dtype=loader.dtype).element_size()
    # Signals that fire through per-layer forward hooks (router_logits only
    # fires on MoE layers, but counting it at the ceiling is fine for a
    # warning).
    hooked = len([s for s in ctx.signals if s in _HOOKED_SIGNALS])

    chunk_size = int(getattr(loader, "_chunk_size", 1))
    capture_bytes = (
        chunk_size
        * max(hooked, 1)
        * batch_size
        * ctx.s_max
        * ctx.hidden_dim
        * dtype_bytes
    )

    acc_bytes = 0
    acc_breakdown: list[tuple[str, int]] = []
    for name, acc in accumulators.items():
        est = getattr(acc, "estimate_bytes", None)
        if est is None:
            continue
        n = int(est(ctx, batch_size))
        acc_bytes += n
        acc_breakdown.append((name, n))

    projected = capture_bytes + acc_bytes
    available = int(psutil.virtual_memory().available)
    if projected < 0.9 * available:
        return

    suggested_cs = max(1, chunk_size // 2)
    breakdown_lines = [
        f"  capture (chunk_size={chunk_size} × {max(hooked, 1)} signals "
        f"× batch={batch_size} × s_max={ctx.s_max} × hidden="
        f"{ctx.hidden_dim} × {dtype_bytes}B): {_fmt_gib(capture_bytes)}",
    ]
    for name, n in acc_breakdown:
        breakdown_lines.append(f"  accumulator {name!r}: {_fmt_gib(n)}")

    import warnings
    warnings.warn(
        "lmprobe.sweep: projected new RAM allocations "
        f"({_fmt_gib(projected)}) approach or exceed 90% of system-"
        f"available ({_fmt_gib(available)}). The OOM killer will SIGKILL "
        f"the process without a Python traceback if the sweep exceeds "
        f"RAM. Breakdown:\n"
        + "\n".join(breakdown_lines)
        + (
            f"\nTry chunk_size={suggested_cs} on the backend, or reduce "
            f"signals / batch_size."
            if chunk_size > 1
            else "\nTry reducing signals / batch_size."
        ),
        category=UserWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Driver (skeleton; wiring to backends lands in Task #3)
# ---------------------------------------------------------------------------


def sweep(
    prompts: list[str] | PreTokenizedPrompts,
    *,
    accumulators: Mapping[str, Accumulator],
    loader: LayerLoader,
    external_bases: Mapping[str, np.ndarray] | None = None,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Run a single-pass sweep over ``prompts``, dispatching signals to
    ``accumulators``.

    Parameters
    ----------
    prompts : list[str]
        Corpus of prompts to forward.
    accumulators : mapping of name -> Accumulator
        Accumulators to run. Signals are derived from the union of their
        ``signals`` attributes.
    loader : LayerLoader
        Backend strategy — owns layer weight lifecycle, embedding, rotary,
        residual buffer, and lm_head application.
    external_bases : mapping of sig -> np.ndarray or None
        Pre-fit PCA bases, shape ``[n_layers, dim, k]`` per signal. Required
        whenever a projection-wanting accumulator subscribes to that signal.
    batch_size : int
        Microbatch size for the forward pass.

    Returns
    -------
    dict[str, Any]
        ``{accumulator_name: accumulator.finalize()}`` — the union of all
        accumulator outputs, keyed by their name in ``accumulators``.

    Notes
    -----
    The actual layer forward loop lives in the backend via ``loader``.
    This skeleton establishes the contract; wiring lands with
    ``ChunkedLayerLoader`` (spec 004 memmap residuals) and
    ``DiskOffloadLayerLoader`` (per-layer safetensors materialization).
    """
    # 1. Resolve signal subscriptions + dispatch plans.
    signals = _collect_signals(accumulators)
    plans = _resolve_plans(accumulators, loader.num_layers, external_bases)

    # 2. Tokenize + embed + compute rotary; allocate residual buffer.
    with loader.prepare(prompts, batch_size=batch_size) as state:
        # 3. Build SweepContext and init accumulators.
        signal_dims: dict[str, int] = {}
        k_per_sig: dict[str, list[int]] = {}
        if external_bases is not None:
            for sig, arr in external_bases.items():
                signal_dims[sig] = int(arr.shape[1])
                k_per_sig[sig] = [int(arr.shape[2])] * int(arr.shape[0])

        ctx = SweepContext(
            n_samples=len(prompts),
            num_layers=loader.num_layers,
            signals=signals,
            signal_dims=signal_dims,
            hidden_dim=state.hidden_dim,
            dtype=loader.dtype,
            device=loader.device,
            seq_lengths=state.seq_lengths,
            s_max=int(state.attention_mask.shape[1]),
            k_per_sig=k_per_sig,
        )
        # Pre-flight RAM estimate — catches the silent-OOM-SIGKILL class
        # of failures (issue #275) before we allocate accumulator buffers.
        _preflight_memory_check(loader, ctx, accumulators, batch_size)
        for acc in accumulators.values():
            acc.init(ctx)

        # 4. Pre-load external bases to device (one-time cost; freed at sweep end).
        basis_gpu: dict[str, torch.Tensor] = {}
        if external_bases is not None:
            for sig, arr in external_bases.items():
                basis_gpu[sig] = torch.from_numpy(
                    np.ascontiguousarray(arr),
                ).to(device=loader.device, dtype=torch.float32)

        # 5. Layer-group / batch forward loop. The concrete forward loop
        #    lives on the loader as ``_drive_sweep``.
        _run_layer_sweep(
            loader=loader,
            state=state,
            accumulators=accumulators,
            plans=plans,
            basis_gpu=basis_gpu,
            batch_size=batch_size,
            ctx=ctx,
        )

    # 6. Finalize accumulators, unpack outputs.
    return {name: acc.finalize() for name, acc in accumulators.items()}


def _run_layer_sweep(
    *,
    loader: LayerLoader,
    state: EmbedState,
    accumulators: Mapping[str, Accumulator],
    plans: dict[tuple[int, str], _SignalPlan],
    basis_gpu: dict[str, torch.Tensor],
    batch_size: int,
    ctx: SweepContext,
) -> None:
    """Inner layer-group + batch loop.

    Iterates ``loader.iter_layer_groups()``; for each group, the batched
    forward runs through the group's layers on device, hooks emit signal
    deltas, and :func:`_dispatch_plan` routes each delta to subscribers.
    After the final layer's forward, if any accumulator subscribes to
    ``logits``, ``loader.apply_lm_head`` is invoked and the resulting
    logits dispatched.

    Implementation is provided by ``backends.py`` integration
    (see ``ChunkedLayerLoader._drive_sweep`` / equivalent). This
    indirection keeps ``sweep.py`` backend-agnostic — it never imports
    ``transformers`` or model-specific helpers.
    """
    # Defer to the loader's driver hook. Loaders implement `_drive_sweep`
    # as the concrete forward loop; this keeps sweep.py pure-protocol.
    driver = getattr(loader, "_drive_sweep", None)
    if driver is None:
        raise NotImplementedError(
            f"{type(loader).__name__} does not implement _drive_sweep; the "
            f"layer forward loop is backend-specific and must be supplied "
            f"by the concrete LayerLoader."
        )
    driver(
        state=state,
        accumulators=accumulators,
        plans=plans,
        basis_gpu=basis_gpu,
        batch_size=batch_size,
        ctx=ctx,
    )


__all__ = [
    "Accumulator",
    "EmbedState",
    "LayerLoader",
    "LifecycleAccumulator",
    "SIGNAL_ATTN_DELTA",
    "SIGNAL_LOGITS",
    "SIGNAL_MLP_DELTA",
    "SIGNAL_RESIDUAL",
    "SIGNAL_ROUTER_LOGITS",
    "SweepContext",
    "VALID_SIGNALS",
    "sweep",
]
