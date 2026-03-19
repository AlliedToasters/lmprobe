# Pooling

Functions for aggregating token-level activations into a single representation.

Pooling strategies can be prefixed with `score:` (post-probe) or `activation:` (pre-probe) to control when reduction happens. See [`parse_pooling_strategy`](#lmprobe.pooling.parse_pooling_strategy) for details.

---

::: lmprobe.pooling.parse_pooling_strategy
    options:
      show_root_heading: true
      show_source: false

---

::: lmprobe.pooling.get_pooling_fn
    options:
      show_root_heading: true
      show_source: false

::: lmprobe.pooling.resolve_pooling
    options:
      show_root_heading: true
      show_source: false

::: lmprobe.pooling.pool_last_token
    options:
      show_root_heading: true
      show_source: false

::: lmprobe.pooling.pool_mean
    options:
      show_root_heading: true
      show_source: false

::: lmprobe.pooling.pool_first_token
    options:
      show_root_heading: true
      show_source: false

::: lmprobe.pooling.pool_max
    options:
      show_root_heading: true
      show_source: false

::: lmprobe.pooling.pool_min
    options:
      show_root_heading: true
      show_source: false

::: lmprobe.pooling.pool_all
    options:
      show_root_heading: true
      show_source: false

::: lmprobe.pooling.reduce_scores
    options:
      show_root_heading: true
      show_source: false
