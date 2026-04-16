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

    def __init__(self, scan_dir: str | Path) -> None:
        from . import scan_storage

        self._scan_dir = Path(scan_dir)
        self._metadata = scan_storage.read_metadata(self._scan_dir)
        self._bases = scan_storage.read_basis(self._scan_dir, "0_global")
        assert isinstance(self._bases, dict)
        self._channel_config = scan_storage.read_channel_config(
            self._scan_dir, "0_global",
        )
        self._samples_table = scan_storage.read_samples(self._scan_dir)
        self._projections: np.memmap | None = None
        self._coords_table: Any = None

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

        Returns
        -------
        SampleScan
            The fitted scan, loaded from disk.
        """
        import torch

        from . import scan_storage
        from .backends import ChunkedLocalBackend

        if dtype is None:
            dtype = torch.bfloat16

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        scan_dir = Path(scan_dir)

        backend_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "device": device,
            "dtype": dtype,
        }
        if chunk_size != "auto":
            backend_kwargs["chunk_size"] = chunk_size
        backend = ChunkedLocalBackend(**backend_kwargs)

        # Run the scan forward pass
        (
            metadata_dict,
            bases,
            projections,
            coords,
            token_ids_per_sample,
            seq_lengths,
            _attention_mask,
            signal_dims,
        ) = backend.scan_forward(
            prompts,
            signals=signals,
            n_components=n_components,
            batch_size=batch_size,
            generative_masks=generative_masks,
        )

        actual_signals = metadata_dict["signals"]
        creation_date = datetime.datetime.now(datetime.timezone.utc).isoformat()

        scan_storage.write_metadata(
            scan_dir,
            scan_storage.ScanMetadata(
                model_id=metadata_dict["model_id"],
                hidden_dim=metadata_dict["hidden_dim"],
                n_layers=metadata_dict["n_layers"],
                n_components=metadata_dict["n_components"],
                n_samples=metadata_dict["n_samples"],
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

        # Build signal info for channel config
        signal_info = []
        for sig in actual_signals:
            dim = signal_dims.get(sig, metadata_dict["hidden_dim"])
            k_eff = min(n_components, dim)
            signal_info.append({
                "name": sig,
                "dim": dim,
                "k_effective": k_eff,
            })

        scan_storage.write_channel(
            scan_dir,
            channel_name="0_global",
            bases=bases,
            config={
                "name": "global",
                "k": n_components,
                "fit_method": "pca",
                "signals": signal_info,
            },
        )

        scan_storage.write_projections(
            scan_dir,
            values=projections,
            coords_sample_id=np.array(coords["sample_id"], dtype=np.int32),
            coords_layer=np.array(coords["layer"], dtype=np.int16),
            coords_token_pos=np.array(coords["token_pos"], dtype=np.int16),
            coords_signal=np.array(coords["signal"], dtype=np.int8),
        )

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

    # --- Projection ---

    def _get_backend(self) -> Any:
        """Lazy-load the ChunkedLocalBackend for projection."""
        if self._backend is None:
            import torch

            from .backends import ChunkedLocalBackend

            device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
