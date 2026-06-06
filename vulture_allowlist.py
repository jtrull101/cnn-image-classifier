# Vulture allowlist — names that are dynamically used and must not be flagged as dead code.
# Regenerate additions with: uv run vulture --make-whitelist
# Consumed automatically by `make deadcode` (see [tool.vulture] in pyproject.toml).

# Keras Callback hooks receive `epoch` by API contract even when unused.
epoch  # unused variable (packages/training/img_classifier_training/callbacks.py:26)
epoch  # unused variable (packages/training/img_classifier_training/callbacks.py:64)
