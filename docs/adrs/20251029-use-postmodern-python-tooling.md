# Use Postmodern Python Tooling Stack

## Context and Problem Statement

Chess-sandbox requires a modern Python development environment optimized for a solo developer using AI-assisted coding. The tooling must provide fast, actionable feedback and minimize configuration overhead. What tooling approach should be adopted to optimize developer experience and enable effective AI-aided development workflows?

## Considered Options

* [Postmodern Python](https://rdrn.me/postmodern-python/) approach (uv, ruff, pyright, pytest)
* Traditional setup (pip, venv, flake8, mypy)
* Poetry-based workflow

## Decision Outcome

Chosen option: "[Postmodern Python](https://github.com/carderne/postmodern-python)", because it provides modern, opinionated tooling with fast feedback loops ideal for AI-aided development. The stack delivers immediate, structured feedback that both humans and AI agents can act on quickly, while eliminating configuration bikeshedding common in solo projects.

### Consequences

* Good, because fast feedback loops from modern tools (uv, ruff) enable rapid iteration in AI-assisted workflows
* Good, because opinionated defaults reduce configuration overhead for solo developers
* Good, because structured output from linters and type checkers is easily parsed and acted upon by AI agents
* Good, because significantly faster tool execution compared to traditional alternatives (pip, flake8, mypy)
* Bad, because less widespread adoption means fewer community resources and Stack Overflow answers
* Bad, because newer tooling may have less mature ecosystem integrations
