"""Baseline classifiers for comparison with linear probes.

These baselines help determine whether a probe is learning something
beyond simple lexical features. If a probe doesn't beat bag-of-words,
it's likely just doing token matching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .classifiers import resolve_classifier

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class BaselineProbe:
    """Text classification baseline for comparison with linear probes.

    This class provides simple baselines that don't use model activations,
    helping determine if probes are learning meaningful representations
    or just exploiting surface-level features.

    Parameters
    ----------
    method : str, default="tfidf"
        Feature extraction method:
        - "bow": Bag-of-words (word counts)
        - "tfidf": TF-IDF weighted bag-of-words
        - "random": Random predictions (true chance baseline)
        - "majority": Always predict majority class
        - "sentence_length": Use character/word count features
        - "perplexity": Use model's own logprobs (requires model param)
        - "sentence_transformers": Use off-the-shelf embeddings
    classifier : str | BaseEstimator, default="logistic_regression"
        Classification model. Same options as LinearProbe.
        Ignored for method="random" and method="majority".
    random_state : int | None, default=None
        Random seed for reproducibility.
    max_features : int | None, default=10000
        Maximum vocabulary size for bow/tfidf methods.
    ngram_range : tuple[int, int], default=(1, 1)
        N-gram range for bow/tfidf. (1, 1) = unigrams only,
        (1, 2) = unigrams and bigrams.
    model : str | None, default=None
        HuggingFace model ID. Required for method="perplexity".
    device : str, default="auto"
        Device for model inference (for perplexity method).
    remote : bool, default=False
        Use nnsight remote execution (for perplexity method).

    Attributes
    ----------
    classifier_ : BaseEstimator
        The fitted classifier (after calling fit()).
    classes_ : np.ndarray
        Class labels (after calling fit()).
    vectorizer_ : CountVectorizer | TfidfVectorizer | None
        The fitted text vectorizer (for bow/tfidf methods).

    Examples
    --------
    >>> baseline = BaselineProbe(method="tfidf", classifier="logistic_regression")
    >>> baseline.fit(positive_prompts, negative_prompts)
    >>> accuracy = baseline.score(test_prompts, test_labels)
    >>> print(f"TF-IDF baseline: {accuracy:.1%}")
    """

    VALID_METHODS = (
        "bow",
        "tfidf",
        "random",
        "majority",
        "sentence_length",
        "perplexity",
        "sentence_transformers",
    )

    def __init__(
        self,
        method: str = "tfidf",
        classifier: str | BaseEstimator = "logistic_regression",
        random_state: int | None = None,
        max_features: int | None = 10000,
        ngram_range: tuple[int, int] = (1, 1),
        model: str | None = None,
        device: str = "auto",
        remote: bool = False,
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown method: {method!r}. Valid options: {self.VALID_METHODS}"
            )

        if method == "perplexity" and model is None:
            raise ValueError(
                "method='perplexity' requires the 'model' parameter to be specified."
            )

        self.method = method
        self.classifier = classifier
        self.random_state = random_state
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.model = model
        self.device = device
        self.remote = remote

        # Resolve classifier template
        if method in ("random", "majority"):
            strategy = "uniform" if method == "random" else "most_frequent"
            self._classifier_template = DummyClassifier(
                strategy=strategy, random_state=random_state
            )
        else:
            self._classifier_template = resolve_classifier(classifier, random_state)

        # Fitted state
        self.classifier_: BaseEstimator | None = None
        self.classes_: np.ndarray | None = None
        self.vectorizer_: CountVectorizer | TfidfVectorizer | None = None
        self._st_model = None  # Sentence transformer model (lazy loaded)

    def fit(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
    ) -> "BaselineProbe":
        """Fit the baseline on contrastive examples.

        Parameters
        ----------
        positive_prompts : list[str]
            Examples of the positive class.
        negative_prompts : list[str]
            Examples of the negative class.

        Returns
        -------
        BaselineProbe
            Self, for method chaining.
        """
        # Combine prompts and create labels
        prompts = positive_prompts + negative_prompts
        labels = np.array([1] * len(positive_prompts) + [0] * len(negative_prompts))

        # Extract features based on method
        if self.method == "bow":
            self.vectorizer_ = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
            )
            X = self.vectorizer_.fit_transform(prompts)
        elif self.method == "tfidf":
            self.vectorizer_ = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
            )
            X = self.vectorizer_.fit_transform(prompts)
        elif self.method == "sentence_length":
            X = self._extract_sentence_length_features(prompts)
        elif self.method == "perplexity":
            X = self._compute_perplexity(prompts)
        elif self.method == "sentence_transformers":
            X = self._transform_sentence_transformers(prompts, fit=True)
        else:
            # random/majority - features don't matter, just need shape
            X = np.zeros((len(prompts), 1))

        # Fit classifier
        self.classifier_ = clone(self._classifier_template)
        # Some classifiers (LDA) don't support sparse matrices
        X = self._ensure_dense_if_needed(X)
        self.classifier_.fit(X, labels)
        self.classes_ = self.classifier_.classes_

        return self

    def predict(self, prompts: list[str]) -> np.ndarray:
        """Predict class labels for prompts.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to classify.

        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_prompts,).
        """
        self._check_fitted()
        X = self._transform(prompts)
        return self.classifier_.predict(X)

    def predict_proba(self, prompts: list[str]) -> np.ndarray:
        """Predict class probabilities for prompts.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to classify.

        Returns
        -------
        np.ndarray
            Class probabilities, shape (n_prompts, n_classes).

        Raises
        ------
        AttributeError
            If the classifier doesn't support predict_proba.
        """
        self._check_fitted()

        if not hasattr(self.classifier_, "predict_proba"):
            raise AttributeError(
                f"Classifier {type(self.classifier_).__name__} does not support "
                "predict_proba(). Use a classifier like logistic_regression or lda."
            )

        X = self._transform(prompts)
        return self.classifier_.predict_proba(X)

    def score(self, prompts: list[str], labels: list[int] | np.ndarray) -> float:
        """Compute classification accuracy.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to classify.
        labels : list[int] | np.ndarray
            True labels.

        Returns
        -------
        float
            Classification accuracy.
        """
        self._check_fitted()
        predictions = self.predict(prompts)
        labels = np.asarray(labels)
        return float(np.mean(predictions == labels))

    def _transform(self, prompts: list[str]) -> np.ndarray:
        """Transform prompts to feature matrix."""
        if self.method in ("bow", "tfidf"):
            X = self.vectorizer_.transform(prompts)
            return self._ensure_dense_if_needed(X)
        elif self.method == "sentence_length":
            return self._extract_sentence_length_features(prompts)
        elif self.method == "perplexity":
            return self._compute_perplexity(prompts)
        elif self.method == "sentence_transformers":
            return self._transform_sentence_transformers(prompts, fit=False)
        else:
            # random/majority
            return np.zeros((len(prompts), 1))

    def _ensure_dense_if_needed(self, X: np.ndarray) -> np.ndarray:
        """Convert sparse matrix to dense if classifier requires it."""
        if issparse(X) and isinstance(self._classifier_template, LinearDiscriminantAnalysis):
            return X.toarray()
        return X

    def _extract_sentence_length_features(self, prompts: list[str]) -> np.ndarray:
        """Extract sentence length features from prompts.

        Features: character count, word count, average word length.
        """
        features = []
        for prompt in prompts:
            char_count = len(prompt)
            word_count = len(prompt.split())
            avg_word_len = char_count / max(word_count, 1)
            features.append([char_count, word_count, avg_word_len])
        return np.array(features)

    def _compute_perplexity(self, prompts: list[str]) -> np.ndarray:
        """Compute perplexity features for each prompt.

        Uses the model's own logprobs to compute perplexity.
        Features: mean perplexity, min perplexity, max perplexity.
        """
        import torch

        from .extraction import configure_remote, get_cached_model

        if self.remote:
            configure_remote()

        model = get_cached_model(self.model, self.device, remote=self.remote)

        features = []
        for prompt in prompts:
            # Tokenize
            inputs = model.tokenizer(prompt, return_tensors="pt")

            # Forward pass to get logits using nnsight tracing
            with model.trace(inputs, remote=self.remote) as tracer:
                logits = model.lm_head.output.save()

            # Unwrap proxy if needed
            logits_val = logits.value if hasattr(logits, "value") else logits

            # Compute per-token cross-entropy loss
            # Shift logits and labels for next-token prediction
            shift_logits = logits_val[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()

            # Move to same device
            if shift_logits.device != shift_labels.device:
                shift_labels = shift_labels.to(shift_logits.device)

            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Compute perplexity statistics
            mean_loss = per_token_loss.mean().item()
            min_loss = per_token_loss.min().item()
            max_loss = per_token_loss.max().item()

            mean_ppl = float(np.exp(mean_loss))
            min_ppl = float(np.exp(min_loss))
            max_ppl = float(np.exp(max_loss))

            features.append([mean_ppl, min_ppl, max_ppl])

        return np.array(features)

    def _check_sentence_transformers_installed(self) -> None:
        """Check that sentence-transformers is installed."""
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for method='sentence_transformers'. "
                "Install it with: pip install lmprobe[embeddings]"
            )

    def _transform_sentence_transformers(
        self, prompts: list[str], fit: bool = False
    ) -> np.ndarray:
        """Extract sentence-transformer embeddings.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to embed.
        fit : bool
            If True, load the sentence transformer model.

        Returns
        -------
        np.ndarray
            Embeddings with shape (n_prompts, embedding_dim).
        """
        self._check_sentence_transformers_installed()
        from sentence_transformers import SentenceTransformer

        if fit or self._st_model is None:
            # Use a small, fast model by default
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = self._st_model.encode(prompts, show_progress_bar=False)
        return embeddings

    def _check_fitted(self) -> None:
        """Check if the baseline has been fitted."""
        if self.classifier_ is None:
            raise RuntimeError(
                "BaselineProbe has not been fitted. Call fit() first."
            )

    def get_feature_names(self) -> list[str] | None:
        """Get feature names for bow/tfidf methods.

        Returns
        -------
        list[str] | None
            Feature names (vocabulary), or None for random/majority.
        """
        if self.vectorizer_ is None:
            return None
        return self.vectorizer_.get_feature_names_out().tolist()

    def get_top_features(self, n: int = 20) -> dict[str, list[tuple[str, float]]] | None:
        """Get top features by classifier weight for each class.

        Only works for bow/tfidf with linear classifiers that have coef_.

        Parameters
        ----------
        n : int, default=20
            Number of top features to return per class.

        Returns
        -------
        dict | None
            Dictionary with 'positive' and 'negative' keys, each containing
            a list of (feature_name, weight) tuples. None if not applicable.
        """
        if self.vectorizer_ is None or self.classifier_ is None:
            return None

        if not hasattr(self.classifier_, "coef_"):
            return None

        feature_names = self.get_feature_names()
        coef = self.classifier_.coef_.ravel()

        # Get indices sorted by weight
        top_positive_idx = np.argsort(coef)[-n:][::-1]
        top_negative_idx = np.argsort(coef)[:n]

        return {
            "positive": [(feature_names[i], coef[i]) for i in top_positive_idx],
            "negative": [(feature_names[i], coef[i]) for i in top_negative_idx],
        }
