# Standardize on httpx for HTTP Client and respx for Testing

## Context and Problem Statement

The codebase currently uses mixed HTTP libraries: `requests` in `lichess_export.py` for API calls and `httpx` (transitively via the OpenAI SDK) in `refiner.py`. Additionally, unit tests for the Refiner class require real OpenAI API tokens, making them slow, non-deterministic, and dependent on external services. We need a consistent HTTP client library and a modern testing approach that enables fast, reliable unit tests without external dependencies.

## Considered Options

* Keep `requests` + add `requests-mock` for testing
* Standardize on `httpx` + add `respx` for testing
* Standardize on `httpx` + use only `unittest.mock` for testing
* Keep mixed approach with both libraries

## Decision Outcome

Chosen option: "Standardize on `httpx` + add `respx` for testing", because:

1. **Consistency**: The OpenAI SDK (v1.68.2) already uses `httpx` internally, making it the de facto HTTP library in the codebase
2. **Modern features**: `httpx` provides async/sync support, HTTP/2, better type hints, and is actively maintained
3. **Better testing**: `respx` is purpose-built for mocking `httpx` requests with type-safe, pytest-friendly APIs
4. **Simpler dependency tree**: Using one HTTP library reduces complexity compared to maintaining both `requests` and `httpx`
5. **Migration simplicity**: Only one file (`lichess_export.py`) uses `requests` with a simple POST call that's trivial to migrate

### Consequences

* Good, because we have a single, modern HTTP client library across the codebase
* Good, because tests no longer require API keys and run faster without network calls
* Good, because `respx` provides realistic HTTP-layer mocking that tests actual SDK behavior
* Good, because `httpx` has better async support for future scalability needs
* Good, because type hints and modern Python features improve code quality
* Bad, because we add a new testing dependency (`respx`) to the project
* Bad, because developers need to learn `httpx` API if only familiar with `requests` (though APIs are very similar)
* Neutral, because the migration effort is minimal given the limited use of `requests`
