# 004: Classifier Interface

**Status**: Proposed
**Date**: 2026-01-02
**Author**: Claude (drafted), Toast (review pending)

## Context

`lmprobe` trains classifiers on pooled activations. We need to:
1. Provide sensible built-in classifiers
2. Support custom sklearn-compatible estimators
3. Handle the high dimensionality of LLM activations (4096+ dims, potentially concatenated across layers)

## Decision

### Classifier Parameter

The `classifier` parameter accepts either a string (built-in) or an sklearn-compatible estimator:

```python
# Built-in (string)
probe = LinearProbe(classifier="logistic_regression")
probe = LinearProbe(classifier="svm")
probe = LinearProbe(classifier="ridge")

# Custom sklearn estimator
from sklearn.svm import SVC
probe = LinearProbe(classifier=SVC(kernel="linear", probability=True))

# Custom with pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
probe = LinearProbe(
    classifier=Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ])
)
```

### Built-In Classifiers

| String | Implementation | Notes |
|--------|---------------|-------|
| `"logistic_regression"` | `LogisticRegression(max_iter=1000, solver="lbfgs")` | Default, fast, interpretable |
| `"logistic_regression_cv"` | `LogisticRegressionCV(cv=5, max_iter=1000)` | Auto-tuned regularization |
| `"ridge"` | `RidgeClassifier()` | Fast for high-dim, no probabilities |
| `"svm"` | `SVC(kernel="linear", probability=True)` | Linear SVM with Platt scaling |
| `"sgd"` | `SGDClassifier(loss="log_loss")` | Scalable to large datasets |

**Default**: `"logistic_regression"` — fast, supports `predict_proba()`, well-understood.

### sklearn Compatibility Requirements

Custom classifiers must implement the sklearn estimator interface:

```python
class Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    # Optional but recommended
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def score(self, X: np.ndarray, y: np.ndarray) -> float: ...
```

At `__init__` time, we validate:
1. Classifier has `fit` and `predict` methods
2. Warn if `predict_proba` is missing (some users need probabilities)

```python
# Warning: classifier lacks predict_proba
probe = LinearProbe(classifier=RidgeClassifier())
# UserWarning: RidgeClassifier does not support predict_proba().
# probe.predict_proba() will raise an error.
```

### Probability Calibration

For classifiers without native probability support, users can wrap with `CalibratedClassifierCV`:

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

probe = LinearProbe(
    classifier=CalibratedClassifierCV(LinearSVC(), cv=3)
)
```

We don't do this automatically to avoid surprising behavior and computational overhead.

### Handling High Dimensionality

LLM activations are high-dimensional (e.g., 4096 for Llama-8B, 12288 if concatenating 3 layers). Recommendations:

1. **Regularization** (default for `"logistic_regression"`): L2 regularization handles collinearity
2. **Dimensionality reduction**: Users can include PCA in a pipeline
3. **Feature scaling**: Not required for logistic regression on activations (already normalized-ish), but users can add `StandardScaler` if needed

```python
# High-dim setup with explicit handling
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

probe = LinearProbe(
    layers="all",  # 32 layers × 4096 = 131,072 dims
    classifier=Pipeline([
        ("pca", PCA(n_components=1000)),
        ("clf", LogisticRegression()),
    ])
)
```

### Classifier Cloning

We clone the classifier at `fit()` time to ensure:
1. Multiple calls to `fit()` start fresh
2. The original estimator object isn't mutated

```python
from sklearn.base import clone

def fit(self, positive_prompts, negative_prompts):
    # ... extract activations ...
    self._classifier = clone(self.classifier)
    self._classifier.fit(X, y)
```

### Accessing the Trained Classifier

After fitting, users can access the trained classifier for inspection:

```python
probe.fit(pos, neg)

# Access coefficients (for interpretability)
coef = probe.classifier_.coef_  # shape: (1, hidden_dim * num_layers)

# Access full sklearn estimator
probe.classifier_.get_params()
```

Convention: `classifier_` (with trailing underscore) indicates fitted state, following sklearn conventions.

### Multi-Class Support

While the primary use case is binary (positive vs negative), we support multi-class for flexibility:

```python
# Standard sklearn signature with multi-class labels
probe.fit(prompts, labels)  # labels: [0, 1, 2, 0, 1, ...]

# predict_proba returns (batch, num_classes)
probs = probe.predict_proba(test)  # shape: (n, 3) for 3 classes
```

Multi-class uses the classifier's native strategy (OvR for logistic regression, etc.).

### Binary Class Ordering

For binary classification, we follow sklearn conventions:
- Class 0: negative class
- Class 1: positive class

In contrastive mode:
```python
probe.fit(positive_prompts, negative_prompts)
# positive_prompts → label 1
# negative_prompts → label 0

probs = probe.predict_proba(test)
# probs[:, 0] = P(negative)
# probs[:, 1] = P(positive)
```

## Implementation Notes

### Random State Propagation

**Decision**: `LinearProbe.random_state` propagates to all built-in classifiers. This ensures reproducibility from a single parameter.

```python
# LinearProbe's random_state flows to the classifier
probe = LinearProbe(
    model="...",
    classifier="logistic_regression",
    random_state=42,  # Propagates to LogisticRegression
)
```

For custom classifiers, users must set `random_state` themselves:
```python
# Custom classifier — user controls random_state
probe = LinearProbe(
    model="...",
    classifier=SVC(kernel="linear", probability=True, random_state=42),
    random_state=42,  # Does NOT automatically propagate to custom classifiers
)
```

### Built-In Classifier Factory

Built-in classifiers are constructed with `random_state` from `LinearProbe`:

```python
def _build_classifier(name: str, random_state: int | None) -> BaseEstimator:
    """Build a classifier with the probe's random_state."""
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        )
    elif name == "logistic_regression_cv":
        return LogisticRegressionCV(
            cv=5,
            max_iter=1000,
            random_state=random_state,
        )
    elif name == "ridge":
        return RidgeClassifier(random_state=random_state)
    elif name == "svm":
        return SVC(
            kernel="linear",
            probability=True,
            random_state=random_state,
        )
    elif name == "sgd":
        return SGDClassifier(
            loss="log_loss",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")
```

### Validation

```python
def _validate_classifier(clf: BaseEstimator) -> None:
    if not hasattr(clf, "fit"):
        raise TypeError("Classifier must have fit() method")
    if not hasattr(clf, "predict"):
        raise TypeError("Classifier must have predict() method")
    if not hasattr(clf, "predict_proba"):
        warnings.warn(
            f"{type(clf).__name__} does not support predict_proba(). "
            "probe.predict_proba() will raise an error.",
            UserWarning,
        )
```

## Alternatives Considered

### Auto-Wrapping with CalibratedClassifierCV
Could automatically wrap classifiers lacking `predict_proba()`. Rejected because:
- Hidden computational cost (CV fitting)
- Surprising behavior
- Users who don't need probabilities shouldn't pay the cost

### Neural Network Classifiers
Could support PyTorch-based classifiers for non-linear probing. Deferred to future work — the "linear" in `LinearProbe` is intentional. Could add `NonlinearProbe` later.

### Ensemble of Layer-Specific Classifiers
Train one classifier per layer, ensemble predictions. More complex, deferred to potential `EnsembleProbe` class.

## Consequences

- **Good**: Full sklearn compatibility — leverage existing ecosystem
- **Good**: Sensible defaults with escape hatches
- **Good**: Built-in classifiers cover common cases
- **Caution**: Users must handle high dimensionality themselves for many-layer setups
- **Caution**: No automatic probability calibration — must document

## References

- scikit-learn estimator interface: https://scikit-learn.org/stable/developers/develop.html
- Logistic regression for probing: Alain & Bengio (2016), Anthropic (2024)
