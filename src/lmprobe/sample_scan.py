"""SampleScan: Breadth-first activation indexing for language models.

A SampleScan captures configurable signals (attention deltas, MLP deltas,
residual stream, router logits) at every layer during a forward pass, fits
PCA on a corpus (the "sample"), and stores a compressed basis per signal.
The basis is a reusable lens: any prompt can be projected through it and
visualized as a 2D RGB image showing what the model is doing at each
(token, layer) coordinate.

Example
-------
>>> from lmprobe import SampleScan
>>>
>>> scan = SampleScan.run(
...     prompts=corpus_prompts,
...     labels=corpus_labels,
...     model_name="Qwen/Qwen2.5-7B-Instruct",
...     scan_dir="./my_scan",
...     signals=["residual", "attn_delta", "mlp_delta"],
... )
>>>
>>> fig = scan.plot("The capital of France is Paris.", signal="attn_delta")
>>> fig.savefig("scan_view.png")
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure


class SampleScan:
    """A compressed activation scan — a reusable PCA basis fit on a corpus.

    The scan stores PCA basis vectors for each (layer, signal) pair,
    plus pre-computed projections for all corpus prompts. The basis can
    be applied to any prompt to produce a per-token, per-layer RGB
    visualization of model internals.

    Parameters
    ----------
    scan_dir : str or Path
        Path to an existing scan directory on disk.
    """

    def __init__(
        self,
        scan_dir: str | Path,
        *,
        backend: str = "auto",
    ) -> None:
        """Load an existing scan from disk.

        Parameters
        ----------
        scan_dir : str or Path
            Path to an existing scan directory.
        backend : str
            Which backend to use for projection forward passes. One of:
              - ``"auto"`` (default): heuristic based on model size vs.
                available CPU RAM. ``disk_offload`` kicks in when the
                estimated model weight footprint exceeds 80% of system RAM.
              - ``"chunked"``: force ``ChunkedLocalBackend`` (full model on
                CPU, layers streamed to GPU in chunks). Fast when the full
                dataset's intermediate residuals + signal captures fit in
                CPU RAM alongside the model weights.
              - ``"disk_offload"``: force ``DiskOffloadBackend`` (layer
                weights streamed directly from safetensors to GPU, never
                resident on CPU). Preferred when the full model wouldn't
                leave room for intermediate activations at the sample
                counts being projected — e.g. fusing many long-sequence
                prompts into one call.
        """
        from . import scan_storage

        self._scan_dir = Path(scan_dir)
        self._metadata = scan_storage.read_metadata(self._scan_dir)
        bases_loaded = scan_storage.read_basis(self._scan_dir, "0_global")
        assert isinstance(bases_loaded, dict)
        self._bases: dict[str, np.ndarray] = bases_loaded
        self._channel_config = scan_storage.read_channel_config(
            self._scan_dir, "0_global",
        )
        self._samples_table = scan_storage.read_samples(self._scan_dir)
        self._projections: np.memmap | None = None
        self._coords_table: Any = None
        # Offset table: [n_samples, n_layers, n_signals, 2] int32.
        # Lazy-loaded on first per_token() / per_token_layer() call.
        # May be None for scans written before the offset table was added
        # — in that case per_token falls back to coord-scanning.
        self._offset_table: np.memmap | None = None
        self._offset_table_checked: bool = False

        if backend not in ("auto", "chunked", "disk_offload"):
            raise ValueError(
                f"backend must be one of 'auto', 'chunked', 'disk_offload'; "
                f"got {backend!r}"
            )
        self._backend_mode = backend

        # Backend for projection forward passes (lazy-loaded)
        self._backend: Any = None

    @classmethod
    def run(
        cls,
        prompts: list[str],
        labels: list[int],
        model_name: str,
        scan_dir: str | Path,
        *,
        signals: list[str] | None = None,
        n_components: int = 64,
        device: str = "auto",
        dtype: Any = None,
        batch_size: int = 4,
        chunk_size: int | str = "auto",
        generative_masks: list[np.ndarray] | None = None,
        backend: str = "chunked",
        attn_implementation: str = "sdpa",
    ) -> SampleScan:
        """Fit a scan on a corpus and write results to disk.

        Parameters
        ----------
        prompts : list[str]
            Corpus prompts to fit the PCA basis on.
        labels : list[int]
            Per-prompt labels (e.g. 0/1 for binary contrast).
        model_name : str
            HuggingFace model ID.
        scan_dir : str or Path
            Directory to write scan artifacts to.
        signals : list[str] or None
            Signals to capture. Defaults to ["attn_delta", "mlp_delta"].
            Valid: "residual", "attn_delta", "mlp_delta", "router_logits".
        n_components : int
            Max PCA components per (layer, signal).
        device : str
            Compute device ("auto", "cpu", "cuda:0", etc.).
        dtype : torch.dtype or None
            Model dtype. Defaults to bfloat16.
        batch_size : int
            Number of prompts per forward-pass batch.
        chunk_size : int or str
            Layers per GPU chunk ("auto" or explicit int).
        generative_masks : list of np.ndarray or None
            Per-sample boolean masks, shape [seq_len_i]. True = generative
            (assistant) token. PCA is fit only on generative tokens to
            avoid prompt leakage. All tokens are still projected.
        backend : str
            Backend to use: "chunked" (loads full model on CPU, streams
            layers through GPU) or "disk_offload" (loads layer weights
            directly from safetensors to GPU — for models exceeding CPU RAM).
        attn_implementation : str
            Attention kernel passed to ``from_pretrained`` for the chunked
            backend. Default ``"sdpa"``. Use ``"eager"`` only if a custom
            signal hook needs materialized ``[B, H, T, T]`` attention weights.
            Ignored by the ``disk_offload`` backend.

        Returns
        -------
        SampleScan
            The fitted scan, loaded from disk.
        """
        import torch

        from . import scan_storage

        if dtype is None:
            dtype = torch.bfloat16

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        scan_dir = Path(scan_dir)

        backend_obj: Any
        if backend == "disk_offload":
            from .backends import DiskOffloadBackend
            backend_obj = DiskOffloadBackend(
                model_name=model_name,
                device=device,
                dtype=dtype,
            )
        else:
            from .backends import ChunkedLocalBackend
            backend_kwargs: dict[str, Any] = {
                "model_name": model_name,
                "device": device,
                "dtype": dtype,
                "attn_implementation": attn_implementation,
            }
            if chunk_size != "auto":
                backend_kwargs["chunk_size"] = chunk_size
            backend_obj = ChunkedLocalBackend(**backend_kwargs)

        # Two-sweep run: PCAFit to derive bases, then PerTokenProjection
        # to project every token through them. Both sweeps share the same
        # loader so weight-loading + rotary-embedding work is paid once per
        # sweep (not once per accumulator).
        from .accumulators import PCAFit, PerTokenProjection
        from .backends import ChunkedLayerLoader, ChunkedLocalBackend
        from .sweep import sweep as _sweep

        signals_list = signals or ["attn_delta", "mlp_delta"]

        if not isinstance(backend_obj, ChunkedLocalBackend):
            raise NotImplementedError(
                "SampleScan.run currently supports ChunkedLocalBackend "
                "only. DiskOffload integration is pending a sweep port "
                "of DiskOffloadBackend."
            )

        # Tokenize once so ids + seq_lengths are available for metadata
        # regardless of what the loader discards internally.
        tokenizer = backend_obj.tokenizer
        tok_out = tokenizer(prompts, padding=True, return_tensors="pt")
        all_input_ids = tok_out["input_ids"]
        all_attention_mask = tok_out["attention_mask"]
        token_ids_per_sample = [
            all_input_ids[i, : int(all_attention_mask[i].sum().item())].tolist()
            for i in range(len(prompts))
        ]
        seq_lengths = [
            int(all_attention_mask[i].sum().item()) for i in range(len(prompts))
        ]

        # Sweep 1: fit PCA bases.
        loader = ChunkedLayerLoader(backend_obj)
        fit_out = _sweep(
            prompts,
            accumulators={
                "fit": PCAFit(
                    signals=signals_list,
                    n_components=n_components,
                    generative_masks=generative_masks,
                ),
            },
            loader=loader,
            batch_size=batch_size,
        )
        bases: dict[str, np.ndarray] = fit_out["fit"]

        # Sweep 2: project every token through the fit bases.
        loader = ChunkedLayerLoader(backend_obj)
        proj_out = _sweep(
            prompts,
            accumulators={"proj": PerTokenProjection(bases)},
            loader=loader,
            external_bases=bases,
            batch_size=batch_size,
        )["proj"]

        projections: np.ndarray = proj_out["values"]
        coords: dict[str, np.ndarray] = proj_out["coords"]
        offset_table: np.ndarray = proj_out["offset_table"]
        # PerTokenProjection sorts signals alphabetically for a stable
        # coords/offset-table ordering; mirror that when writing metadata.
        actual_signals: list[str] = proj_out["signal_names"]

        hidden_dim = loader.hidden_dim
        n_layers = loader.num_layers
        signal_dims: dict[str, int] = {
            s: int(bases[s].shape[1]) for s in actual_signals
        }

        creation_date = datetime.datetime.now(datetime.timezone.utc).isoformat()

        scan_storage.write_metadata(
            scan_dir,
            scan_storage.ScanMetadata(
                model_id=model_name,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_components=n_components,
                n_samples=len(prompts),
                creation_date=creation_date,
                signals=actual_signals,
            ),
        )

        scan_storage.write_samples(
            scan_dir,
            sample_ids=list(range(len(prompts))),
            prompts=prompts,
            labels=labels,
            token_ids_list=token_ids_per_sample,
            seq_lengths=seq_lengths,
        )

        signal_info = [
            {
                "name": sig,
                "dim": signal_dims[sig],
                "k_effective": min(n_components, signal_dims[sig]),
            }
            for sig in actual_signals
        ]

        scan_storage.write_channel(
            scan_dir,
            channel_name="0_global",
            bases={s: bases[s] for s in actual_signals},
            config={
                "name": "global",
                "k": n_components,
                "fit_method": "pca",
                "signals": signal_info,
            },
        )

        # PerTokenProjection gives [total_rows, k_max]; storage format is
        # [N_total, n_channels=1, k] (single channel for SampleScan today).
        if projections.ndim == 2:
            projections = projections[:, None, :]
        scan_storage.write_projections(
            scan_dir,
            values=projections,
            coords_sample_id=coords["sample_id"],
            coords_layer=coords["layer"],
            coords_token_pos=coords["token_pos"],
            coords_signal=coords["signal"],
        )
        scan_storage.write_offset_table(scan_dir, offset_table)

        return cls(scan_dir)

    # --- Properties ---

    @property
    def metadata(self) -> dict[str, Any]:
        """Scan metadata as a dict."""
        from dataclasses import asdict

        return asdict(self._metadata)

    @property
    def n_layers(self) -> int:
        return self._metadata.n_layers

    @property
    def n_samples(self) -> int:
        return self._metadata.n_samples

    @property
    def n_components(self) -> int:
        return self._metadata.n_components

    @property
    def signals(self) -> list[str]:
        """List of signal names in this scan."""
        return self._metadata.signals

    @property
    def bases(self) -> dict[str, np.ndarray]:
        """PCA bases per signal. {signal_name: [n_layers, dim, k_eff]}."""
        return self._bases

    @property
    def samples(self) -> Any:
        """Samples table (pyarrow Table)."""
        return self._samples_table

    # --- Lazy-loaded projection data ---

    def _load_projections(self) -> None:
        if self._projections is None:
            from . import scan_storage

            self._projections = scan_storage.open_projections(self._scan_dir)
            self._coords_table = scan_storage.read_coords(self._scan_dir)

    def _load_offset_table(self) -> np.memmap | None:
        """Lazy-load the offset table. ``None`` for pre-offset-table scans."""
        if not self._offset_table_checked:
            from . import scan_storage
            self._offset_table = scan_storage.open_offset_table(self._scan_dir)
            self._offset_table_checked = True
        return self._offset_table

    # --- New sweep-based public API (per-token as first-class product) ---

    def sweep(
        self,
        prompts: list[str],
        *,
        accumulators: Mapping[str, Any],
        external_bases: dict[str, np.ndarray] | None = None,
        batch_size: int = 4,
    ) -> dict[str, Any]:
        """Run a single sweep over ``prompts`` with ``accumulators``.

        The primary programmatic entry point post-deprecation of
        ``batch_project`` / ``batch_project_grouped`` / ``batch_project_reduced``.
        Pass any combination of accumulators from :mod:`lmprobe.accumulators`
        (e.g. ``PerTokenProjection``, ``LastTokenReducer``, ``MeanReducer``,
        ``HiddenStateCapture``, ``LogitCapture``) or your own.

        Parameters
        ----------
        prompts : list[str]
            Prompts to forward.
        accumulators : mapping of {name: Accumulator}
            Accumulators to drive off the sweep. Output is keyed by name.
        external_bases : dict or None
            External bases for stream-projection. Defaults to
            ``self.bases`` (this scan's fit bases) — suitable for any
            projection-wanting accumulator in ``accumulators``.
        batch_size : int
            Microbatch size.

        Returns
        -------
        dict
            ``{name: accumulator.finalize()}``.
        """
        from .backends import ChunkedLayerLoader, ChunkedLocalBackend
        from .sweep import sweep as _sweep

        if external_bases is None:
            external_bases = self._bases

        backend = self._get_backend()
        if isinstance(backend, ChunkedLocalBackend):
            loader = ChunkedLayerLoader(backend)
        else:
            raise NotImplementedError(
                "SampleScan.sweep() currently supports ChunkedLocalBackend "
                "only; DiskOffloadBackend sweep integration is pending. "
                "For DiskOffload, continue to use batch_project* (DeprecationWarning)."
            )

        return _sweep(
            prompts,
            accumulators=accumulators,
            loader=loader,
            external_bases=external_bases,
            batch_size=batch_size,
        )

    def per_token(
        self,
        sample_id: int,
        layer: int,
        signal: str | None = None,
    ) -> np.ndarray:
        """O(1) slice of per-token projections for one (sample, layer).

        Backed by the offset table written at scan time. Falls back to
        the coord-scanning path for scans created before the offset table
        existed — same result, O(N_rows) cost.

        Parameters
        ----------
        sample_id : int
            Index into the corpus.
        layer : int
            Layer index.
        signal : str or None
            If given, return projections for this signal only: shape
            ``[seq_len, k]``. Else stack all signals: ``[seq_len, n_sig, k]``.

        Returns
        -------
        np.ndarray
            Real tokens only (padding excluded via the offset table's
            end-row bound = ``start + seq_length[sample_id]``).
        """
        offset_table = self._load_offset_table()
        if offset_table is None:
            # Legacy fallback — for scans written pre-offset-table. Trim
            # ``get_projections`` (which sizes by ``token_pos.max()+1`` ==
            # tokenizer-padded length) to the sample's real token count so
            # callers see the same shape as the O(1) offset-table path.
            seq_len = int(
                self._samples_table.column("seq_length").to_pylist()[sample_id]
            )
            dense = self.get_projections(sample_id, signal)
            # dense is [padded_seq_len, n_layers, n_sig, k] or [..., 1, k]
            out = dense[:seq_len, layer, :, :]
            if signal is not None:
                return out[:, 0, :]
            return out

        self._load_projections()
        assert self._projections is not None
        signal_names = self.signals

        if signal is not None:
            si = signal_names.index(signal)
            start, end = offset_table[sample_id, layer, si]
            rows: np.ndarray = np.asarray(
                self._projections[int(start) : int(end), 0, :]
            )
            return rows

        slices: list[np.ndarray] = []
        for si in range(len(signal_names)):
            start, end = offset_table[sample_id, layer, si]
            slices.append(
                np.asarray(self._projections[int(start) : int(end), 0, :])
            )
        return np.stack(slices, axis=1)  # [seq_len, n_sig, k]

    def per_token_layer(
        self,
        layer: int,
        signal: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All real-token projections at ``(layer, signal)`` across the corpus.

        Returns
        -------
        values : np.ndarray
            ``[total_real_tokens, k]`` fp16.
        sample_ids : np.ndarray
            ``[total_real_tokens]`` int32 — which sample each row belongs to.
        token_positions : np.ndarray
            ``[total_real_tokens]`` int16 — position within the sample.
        """
        offset_table = self._load_offset_table()
        if offset_table is None:
            raise RuntimeError(
                "per_token_layer() requires an offset table. This scan was "
                "written before the offset table was introduced; re-run "
                "SampleScan.run() to regenerate, or use the slower "
                "get_projections() path per sample."
            )

        self._load_projections()
        assert self._projections is not None
        signal_names = self.signals
        si = signal_names.index(signal)

        n_samples = self._metadata.n_samples
        all_rows: list[np.ndarray] = []
        all_sids: list[np.ndarray] = []
        all_toks: list[np.ndarray] = []
        for sid in range(n_samples):
            start, end = offset_table[sid, layer, si]
            length = int(end) - int(start)
            if length <= 0:
                continue
            rows = np.asarray(
                self._projections[int(start) : int(end), 0, :]
            )
            all_rows.append(rows)
            all_sids.append(np.full(length, sid, dtype=np.int32))
            all_toks.append(np.arange(length, dtype=np.int16))
        if not all_rows:
            k = self._metadata.n_components
            return (
                np.zeros((0, k), dtype=np.float16),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int16),
            )
        return (
            np.concatenate(all_rows, axis=0),
            np.concatenate(all_sids, axis=0),
            np.concatenate(all_toks, axis=0),
        )

    # --- Query API ---

    def get_projections(
        self,
        sample_id: int,
        signal: str | None = None,
    ) -> np.ndarray:
        """Get stored projections for a corpus sample.

        Parameters
        ----------
        sample_id : int
            Index of the sample in the corpus.
        signal : str or None
            If given, filter to this signal only.

        Returns
        -------
        np.ndarray
            Shape [seq_len, n_layers, n_signals, k].
        """
        self._load_projections()
        assert self._projections is not None
        assert self._coords_table is not None

        coords = self._coords_table
        sample_ids = coords.column("sample_id").to_numpy()
        coord_mask = sample_ids == sample_id

        if signal is not None:
            sig_idx = self.signals.index(signal)
            signal_col = coords.column("signal").to_numpy()
            coord_mask = coord_mask & (signal_col == sig_idx)

        rows = self._projections[coord_mask]  # [N_matching, 1, k]
        values = rows[:, 0, :]  # [N_matching, k]

        layers = coords.column("layer").to_numpy()[coord_mask]
        token_pos = coords.column("token_pos").to_numpy()[coord_mask]
        signal_col = coords.column("signal").to_numpy()[coord_mask]

        n_layers = self._metadata.n_layers
        n_signals = len(self.signals) if signal is None else 1
        seq_len = int(token_pos.max()) + 1 if len(token_pos) > 0 else 0
        k = values.shape[1] if len(values) > 0 else self._metadata.n_components

        if signal is not None:
            result = np.zeros((seq_len, n_layers, 1, k), dtype=np.float32)
            result[token_pos, layers, 0, :] = values.astype(np.float32)
        else:
            result = np.zeros((seq_len, n_layers, n_signals, k), dtype=np.float32)
            result[token_pos, layers, signal_col, :] = values.astype(np.float32)

        return result

    def separability_map(
        self,
        labels: np.ndarray | list[int] | None = None,
        signal: str | None = None,
        token_positions: np.ndarray | list[int] | None = None,
    ) -> np.ndarray:
        """Compute per-(layer, signal) separability scores.

        Uses one token position per sample and computes max-over-PCs
        AUROC for binary separation.

        Parameters
        ----------
        labels : array-like or None
            Binary labels per sample. If None, uses stored labels.
        signal : str or None
            If given, only compute for this signal.
        token_positions : array-like or None
            Per-sample token position to use for the separability
            computation. Shape [n_samples]. If None, uses the last
            token of each sample (seq_length - 1). Use this to
            restrict analysis to specific regions (e.g. last token
            of the assistant turn) and avoid prompt leakage.

        Returns
        -------
        np.ndarray
            Shape [n_layers, n_signals] of AUROC scores.
        """
        from sklearn.metrics import roc_auc_score

        if labels is None:
            labels = self._samples_table.column("label").to_numpy()
        labels = np.asarray(labels)

        self._load_projections()
        assert self._projections is not None
        assert self._coords_table is not None

        n_layers = self._metadata.n_layers
        k = self._metadata.n_components
        n_samples = self._metadata.n_samples
        sigs = [signal] if signal else self.signals
        sig_indices = [self.signals.index(s) for s in sigs]

        seq_lengths = self._samples_table.column("seq_length").to_numpy()

        if token_positions is not None:
            tok_pos_per_sample = np.asarray(token_positions, dtype=np.int64)
        else:
            tok_pos_per_sample = seq_lengths - 1

        coords = self._coords_table
        sample_ids_col = coords.column("sample_id").to_numpy()
        layers_col = coords.column("layer").to_numpy()
        token_pos_col = coords.column("token_pos").to_numpy()
        signal_col = coords.column("signal").to_numpy()

        result = np.zeros((n_layers, len(sigs)), dtype=np.float64)

        for out_idx, sig_idx in enumerate(sig_indices):
            for layer_idx in range(n_layers):
                sample_vecs = np.zeros((n_samples, k), dtype=np.float32)

                for sid in range(n_samples):
                    target_tok = int(tok_pos_per_sample[sid])
                    mask = (
                        (sample_ids_col == sid)
                        & (layers_col == layer_idx)
                        & (token_pos_col == target_tok)
                        & (signal_col == sig_idx)
                    )
                    rows = self._projections[mask]
                    if len(rows) > 0:
                        sample_vecs[sid] = rows[0, 0, :k].astype(np.float32)

                max_auroc = 0.5
                for pc in range(min(k, sample_vecs.shape[1])):
                    pc_vals = sample_vecs[:, pc]
                    if np.std(pc_vals) < 1e-10:
                        continue
                    try:
                        auroc = roc_auc_score(labels, pc_vals)
                        auroc = max(auroc, 1.0 - auroc)
                        max_auroc = max(max_auroc, auroc)
                    except ValueError:
                        continue

                result[layer_idx, out_idx] = max_auroc

        return result

    # --- Single Projection ---

    def _get_backend(self) -> Any:
        """Lazy-load a backend for projection.

        Resolution order:
          1. If ``self._backend_mode`` was set explicitly at construction
             (``"chunked"`` or ``"disk_offload"``), honor that.
          2. Otherwise (``"auto"``), use DiskOffloadBackend when the
             estimated model weight footprint exceeds 80% of available
             CPU RAM; else ChunkedLocalBackend.
        """
        if self._backend is None:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            mode = self._backend_mode
            if mode == "auto":
                # Estimate model weight size from hidden_dim and n_layers.
                # Rough heuristic: 2 bytes/param (bf16), ~12*hidden^2 params/layer.
                est_bytes = (
                    self._metadata.n_layers
                    * 12
                    * (self._metadata.hidden_dim ** 2)
                    * 2
                )
                mem_avail: float
                try:
                    import os
                    mem_avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                except (ValueError, OSError):
                    mem_avail = float("inf")
                mode = "disk_offload" if est_bytes > 0.8 * mem_avail else "chunked"

            if mode == "disk_offload":
                from .backends import DiskOffloadBackend
                self._backend = DiskOffloadBackend(
                    model_name=self._metadata.model_id,
                    device=device,
                    dtype=torch.bfloat16,
                )
            else:
                from .backends import ChunkedLocalBackend
                self._backend = ChunkedLocalBackend(
                    model_name=self._metadata.model_id,
                    device=device,
                    dtype=torch.bfloat16,
                )
        return self._backend

    def project_prompt(
        self,
        prompt: str,
        signal: str | None = None,
    ) -> tuple[np.ndarray, list[str], np.ndarray | None]:
        """Run a forward pass on a prompt and project through the scan basis.

        Parameters
        ----------
        prompt : str
            The prompt to visualize.
        signal : str or None
            If given, only project this signal. Otherwise all signals.

        Returns
        -------
        tuple
            (projections, tokens, log_probs)
            - projections: [seq_len, n_layers, n_signals, max_k] float32
            - tokens: list of decoded token strings
            - log_probs: [seq_len] float32 next-token log-probs, or None
        """
        import torch

        backend = self._get_backend()
        proj_signals = [signal] if signal else self.signals
        proj_bases = {s: self._bases[s] for s in proj_signals if s in self._bases}

        projections, token_ids, logits = backend.project_forward(
            prompt, proj_bases, proj_signals, include_logits=True,
        )

        tokens = [backend.tokenizer.decode(tid) for tid in token_ids]

        log_probs = None
        if logits is not None:
            probs = torch.nn.functional.log_softmax(logits[0].float(), dim=-1)
            input_ids = backend.tokenizer.encode(prompt, return_tensors="pt")[0]
            seq_len = min(len(input_ids), probs.shape[0])

            lp = np.full(seq_len, np.nan, dtype=np.float32)
            for i in range(seq_len - 1):
                next_token = input_ids[i + 1].item()
                lp[i] = probs[i, next_token].item()
            log_probs = lp[:len(token_ids)]

        return projections[:len(token_ids)], tokens, log_probs

    # --- Visualization ---

    def plot(
        self,
        prompt: str,
        *,
        signal: str | None = None,
        show_surprise: bool = True,
        show_stats: bool = True,
        figsize: tuple[float, float] | None = None,
        title: str = "",
        generative_mask: np.ndarray | None = None,
    ) -> matplotlib.figure.Figure:
        """Render the hero figure for a prompt under this scan's lens.

        Parameters
        ----------
        prompt : str
            The prompt to visualize.
        signal : str or None
            Which signal to plot. If None, plots all signals stacked.
        show_surprise : bool
            Show the log-probability surprise strip.
        show_stats : bool
            Show per-layer statistics on the right edge.
        figsize : tuple or None
            Figure size (width, height) in inches.
        title : str
            Optional figure title.
        generative_mask : np.ndarray or None
            Boolean mask shape [seq_len]. True = assistant/generative
            token, False = prompt token. Prompt tokens are grayed out.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .scan_plot import render_scan_figure

        projections, tokens, log_probs = self.project_prompt(prompt, signal)

        # Layer stats: mean energy across tokens per (layer, signal)
        layer_stats = np.sqrt(
            (projections ** 2).sum(axis=3).mean(axis=0)
        )  # [n_layers, n_signals]

        # Determine which signals to show
        show_signals = [signal] if signal else self.signals

        return render_scan_figure(
            projections=projections,
            tokens=tokens,
            signal_names=show_signals,
            log_probs=log_probs if show_surprise else None,
            layer_stats=layer_stats if show_stats else None,
            figsize=figsize,
            title=title,
            generative_mask=generative_mask,
        )
