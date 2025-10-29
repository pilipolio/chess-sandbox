# Use Modal for Serverless Endpoints

## Context and Problem Statement

Chess-sandbox needs serverless deployment for chess engine analysis endpoints and future ML workloads. As a one man side project, the primary driver is developer experience - minimizing operational overhead while maintaining a path to GPU-based training and inference workloads.

## Considered Options

* Modal serverless platform
* AWS Lambda
* GCP Cloud Functions

## Decision Outcome

Chosen option: "Modal serverless platform", because it optimizes for developer experience. Modal provides [simple FastAPI integration](https://modal.com/docs/guide/webhooks), [straightforward image building](https://modal.com/docs/reference/modal.Image) with native uv support, and [minimal CI/CD setup](https://modal.com/docs/guide/continuous-deployment) - eliminating the infrastructure complexity of AWS/GCP while supporting future GPU workloads.

### Consequences

* Good, because developer experience is optimized for solo projects - no IAM roles, API Gateway configs, or complex deployment pipelines to manage
* Good, because local development with `modal serve` provides immediate feedback without Docker complexity
* Good, because native GPU support enables future ML workloads without platform migration
* Good, because uv integration maintains consistency with project toolchain
* Bad, because vendor lock-in to Modal, though Python code remains portable
* Bad, because integration testing is less straightforward than Docker-based approaches
