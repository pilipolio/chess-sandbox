# AI Agent Instructions

## Coding standards

 * code should be self documenting with self-explanatory variable and method names. Only use comment to highlight complexity or non obvious patterns.

## Pre-commit Checks

### Code

Run all checks before committing:
```bash
uv run poe all
```

Individual commands:
```bash
uv run poe fmt     # ruff format
uv run poe lint    # ruff check --fix
uv run poe check   # pyright
uv run poe test    # pytest
```

### Updating documentation

Check for documentation impacted by command line or api changes. Don't modify `CLAUDE.md` unless specifically prompted.

## Tools

- **uv**: Package and environment management
- **ruff**: Formatting and linting
- **pyright**: Type checking (strict mode)
- **pytest**: Testing (with doctests)

## Configuration

See `pyproject.toml` for all tool configurations.
