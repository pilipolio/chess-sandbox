# Use OpenAI SDK Directly Without LLM Frameworks

## Context and Problem Statement

Our chess concept labeling and analysis system requires LLM capabilities for concept validation, commentary generation, and evaluation. We need to decide whether to use a specific AI provider SDK directly or adopt an LLM orchestration framework like LangChain or LlamaIndex. The key considerations are access to proprietary features, cost optimization, development velocity, and architectural simplicity.

## Considered Options

* OpenAI SDK directly (openai>=2.6.1)
* LangChain - Popular LLM application framework with chains and agents
* LlamaIndex - Framework focused on data indexing and retrieval
* Haystack - NLP framework with LLM pipeline support

## Decision Outcome

Chosen option: "OpenAI SDK directly", because it provides direct access to proprietary OpenAI features (Responses API with structured outputs, Batch API for cost optimization), maintains a simpler architecture without unnecessary abstractions, and allows for better cost control and optimization. Our use case doesn't require the complex orchestration capabilities of frameworks, and staying close to the metal enables faster iteration and easier debugging.

### Consequences

* Good, because we have direct access to OpenAI's proprietary features like the Responses API (structured outputs with Pydantic) and Batch API (50% cost savings)
* Good, because the codebase remains simpler without framework abstraction layers, making it easier to understand and debug
* Good, because we maintain full control over API calls, enabling precise cost optimization (e.g., prompt caching, batch processing)
* Good, because we can adopt new OpenAI features immediately without waiting for framework support
* Good, because our current use case (concept validation, commentary, evaluation) doesn't require complex agent orchestration or multi-step chains
* Bad, because we have vendor lock-in to OpenAI and would need significant refactoring to support alternative LLM providers
* Bad, because we lose framework-provided utilities like automatic retries, fallbacks, and prompt templates (though we can implement these ourselves as needed)
* Neutral, because we can always adopt a framework later if our needs evolve toward complex agentic workflows
