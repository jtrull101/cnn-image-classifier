Run tests for: $ARGUMENTS

## Decision logic

**Specific file path given:**
```bash
uv run pytest <path> -v
```

**Package name given** (config, utils, data, models, training, cli, api):
```bash
make test-<package>
```

**"unit":**
```bash
make test-unit
```

**"integration":**
```bash
make test-integration
```

**No argument / "all":**
```bash
make test
```

## Markers

```bash
uv run pytest -m unit          # Fast unit tests
uv run pytest -m smoke         # Critical path only
uv run pytest -m "not slow"    # Skip slow tests
```

## Debugging parallel failures

```bash
uv run pytest <test_file> -n 0   # Disable parallelism
```

Report results including full failure tracebacks.
