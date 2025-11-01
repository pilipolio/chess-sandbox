# Use HuggingFace Hub for Dataset and Model Versioning

## Context and Problem Statement

The chess-sandbox project currently uses manual versioning for models (pickle files with embedded metadata) and datasets (local JSONL files). This approach lacks centralized version control, consumes significant local storage (particularly from lczerolens models), and provides no mechanism for sharing or collaborative development. We already consume the Waterhorse/chess_data dataset from HuggingFace Hub, and need a lightweight experiment tracking solution before investing in a full-fledged platform.

## Considered Options

* Continue with local storage and manual versioning
* Data Version Control (DVC) with remote storage (S3/GCS)
* MLflow for full experiment tracking and model registry
* HuggingFace Hub for datasets, models, and model card metadata

## Decision Outcome

Chosen option: "HuggingFace Hub for datasets, models, and model card metadata", because it provides immediate value with minimal overhead. We already depend on it (via Waterhorse/chess_data dataset), have the library available as a transitive dependency (huggingface-hub v0.36.0), and can leverage model cards for lightweight experiment tracking. The free hosting addresses our storage constraints, and the platform is designed specifically for ML artifacts. Model card metadata serves as a stopgap for experiment tracking until we're ready to adopt a dedicated platform.

### Consequences

* Good, because it reduces local storage usage and provides free remote hosting for datasets and models
* Good, because version control and collaboration become straightforward through the Hub's Git-based infrastructure
* Good, because model cards provide lightweight metadata tracking (training metrics, dataset versions, hyperparameters) without additional tooling
* Good, because we're already using the platform for the Waterhorse/chess_data dataset
* Bad, because migrating existing models and datasets requires upfront effort
* Bad, because model card metadata is limited compared to dedicated experiment tracking platforms (acknowledged as temporary solution)
* Bad, because it creates some ecosystem lock-in, though the Git backend allows for migration if needed
