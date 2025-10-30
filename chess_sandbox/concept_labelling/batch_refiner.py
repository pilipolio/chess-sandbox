"""Batch processing for LLM-based concept validation using OpenAI Batch API."""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any

from openai import OpenAI

from .models import Concept, LabelledPosition
from .refiner import ConceptValidation


@dataclass
class BatchRefiner:
    """Validates chess concepts using OpenAI Batch API for cost-effective processing.

    Provides 50-92% cost reduction compared to synchronous API by:
    - Using Batch API (50% discount)
    - Enabling prompt caching (15-20% additional savings)
    - Supporting efficient models like GPT-5 nano (5x cheaper than mini)

    Typical workflow:
    1. prepare_batch_input() - Generate JSONL batch file
    2. submit_batch() - Upload and create batch job
    3. poll_status() - Monitor progress (completes in <24h, often faster)
    4. download_results() - Retrieve completed batch output
    5. process_results() - Parse and apply validated concepts
    """

    PROMPT_TEMPLATE = dedent("""
        You are a chess expert validating whether a concept applies to a game annotation comment.

        POSITION: {side_to_move} to move
        COMMENT: "{comment}"
        CONCEPT TO VALIDATE: "{concept_name}"

        Determine:

        1. **Is this concept truly discussed in the comment?**
           - FALSE POSITIVE examples: "material" wrongly matched as "mate"
           - Concept detected by regex but not actually discussed

        2. **If VALID, what is the TEMPORAL CONTEXT?**
           - 'actual': Concept exists in the current position NOW
             (e.g., "there is a pin", "A fork, ...", "is a passed pawn")
           - 'future': Concept is threatened/possible in future moves
             (e.g., "threatening mate", "can fork", "will be passed pawns")
           - 'hypothetical': Discussing "if/could/would" scenarios
             (e.g., "if black plays Nf6 then there would be a pin")
           - 'past': Referring to previous moves that already happened
             (e.g., "the pin was broken", "after the fork")

        Be strict: only validate if the concept is clearly mentioned in the comment.
    """).strip()

    model: str
    client: OpenAI
    batch_dir: Path
    enable_caching: bool

    @classmethod
    def create(
        cls, model: str = "gpt-5-nano", batch_dir: Path | None = None, enable_caching: bool = True
    ) -> "BatchRefiner":
        """Create BatchRefiner with default configuration.

        Args:
            model: OpenAI model to use (default: gpt-5-nano for cost-efficiency)
            batch_dir: Directory for batch files (default: data/batches)
            enable_caching: Enable prompt caching for repeated content (default: True)

        Returns:
            Configured BatchRefiner instance
        """
        if batch_dir is None:
            batch_dir = Path("data/batches")

        batch_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            model=model,
            client=OpenAI(),
            batch_dir=batch_dir,
            enable_caching=enable_caching,
        )

    def prepare_batch_input(self, positions: list[LabelledPosition]) -> Path:
        """Generate JSONL batch input file with all validation requests.

        Creates one API request per concept validation with custom_id format:
        "pos_{position_idx}_concept_{concept_idx}"

        Note: Prompt caching is enabled by default for /v1/responses endpoint.
        The repeated prompt template content should be automatically cached by OpenAI,
        providing 15-20% additional cost savings beyond the 50% batch discount.

        Args:
            positions: List of LabelledPosition objects with concepts to validate

        Returns:
            Path to generated JSONL batch input file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = self.batch_dir / f"batch_input_{timestamp}.jsonl"
        mapping_file = self.batch_dir / f"batch_mapping_{timestamp}.json"

        mapping: dict[str, dict[str, Any]] = {}
        requests: list[dict[str, Any]] = []

        for pos_idx, position in enumerate(positions):
            for concept_idx, concept in enumerate(position.concepts):
                custom_id = f"pos_{pos_idx}_concept_{concept_idx}"

                # Store mapping for result processing
                mapping[custom_id] = {
                    "position_idx": pos_idx,
                    "concept_idx": concept_idx,
                    "game_id": position.game_id,
                    "concept_name": concept.name,
                }

                # Build prompt
                prompt = self.PROMPT_TEMPLATE.format(
                    side_to_move=position.side_to_move,
                    comment=position.comment,
                    concept_name=concept.name,
                )

                # Create batch request matching OpenAI Batch API format
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": self.model,
                        "input": prompt,
                        "text_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "ConceptValidation",
                                "strict": True,
                                "schema": ConceptValidation.model_json_schema(),
                            },
                        },
                    },
                }

                requests.append(request)

        # Write JSONL batch file
        with batch_file.open("w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        # Write mapping file
        with mapping_file.open("w") as f:
            json.dump(mapping, f, indent=2)

        return batch_file

    def submit_batch(self, batch_file: Path) -> str:
        """Upload batch file and submit job. Returns batch_id.

        Args:
            batch_file: Path to JSONL batch input file

        Returns:
            Batch job ID for polling and result retrieval
        """
        # Upload batch file
        with batch_file.open("rb") as f:
            file_obj = self.client.files.create(file=f, purpose="batch")

        # Create batch job
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )

        # Save metadata for resumability
        metadata = {
            "batch_id": batch.id,
            "batch_file": str(batch_file),
            "input_file_id": file_obj.id,
            "submitted_at": datetime.now().isoformat(),
            "status": batch.status,
            "model": self.model,
        }

        metadata_file = self.batch_dir / f"batch_{batch.id}.json"
        with metadata_file.open("w") as f:
            json.dump(metadata, f, indent=2)

        return batch.id

    def poll_status(self, batch_id: str) -> dict[str, Any]:
        """Check batch job status and return progress metrics.

        Args:
            batch_id: Batch job ID from submit_batch()

        Returns:
            Dictionary with status, request_counts, and progress info
        """
        batch = self.client.batches.retrieve(batch_id)

        request_counts = batch.request_counts
        status_info = {
            "batch_id": batch.id,
            "status": batch.status,
            "request_counts": {
                "total": request_counts.total if request_counts else 0,
                "completed": request_counts.completed if request_counts else 0,
                "failed": request_counts.failed if request_counts else 0,
            },
            "created_at": batch.created_at,
            "in_progress_at": batch.in_progress_at,
            "completed_at": batch.completed_at if batch.status == "completed" else None,
            "failed_at": batch.failed_at if batch.status == "failed" else None,
            "output_file_id": batch.output_file_id if batch.output_file_id else None,
            "error_file_id": batch.error_file_id if batch.error_file_id else None,
        }

        # Update metadata file with latest status
        metadata_file = self.batch_dir / f"batch_{batch_id}.json"
        if metadata_file.exists():
            with metadata_file.open("r") as f:
                metadata = json.load(f)
            metadata["status"] = batch.status
            metadata["request_counts"] = status_info["request_counts"]
            metadata["output_file_id"] = status_info["output_file_id"]
            if batch.status == "completed":
                metadata["completed_at"] = datetime.now().isoformat()
            with metadata_file.open("w") as f:
                json.dump(metadata, f, indent=2)

        return status_info

    def download_results(self, batch_id: str) -> Path:
        """Download completed batch output file.

        Args:
            batch_id: Batch job ID from submit_batch()

        Returns:
            Path to downloaded JSONL results file
        """
        # Get batch info to retrieve output file ID
        batch = self.client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise ValueError(f"Batch {batch_id} is not completed (status: {batch.status})")

        if not batch.output_file_id:
            raise ValueError(f"Batch {batch_id} has no output file")

        # Download output file content
        file_response = self.client.files.content(batch.output_file_id)

        # Save to local file
        results_file = self.batch_dir / f"batch_output_{batch_id}.jsonl"
        with results_file.open("wb") as f:
            f.write(file_response.read())

        return results_file

    def process_results(self, results_file: Path, original_positions: list[LabelledPosition]) -> list[LabelledPosition]:
        """Parse batch results and update positions with validated concepts.

        Args:
            results_file: Path to JSONL batch output file
            original_positions: Original list of positions (will be updated in-place)

        Returns:
            Updated list of positions with validated concepts
        """
        # Load mapping file to know which custom_id corresponds to which position/concept
        mapping_file = self._find_mapping_file(results_file)
        with mapping_file.open("r") as f:
            mapping = json.load(f)

        # Parse results
        failed_requests: list[dict[str, Any]] = []

        # Create a new concepts list for each position
        new_concepts_by_position: dict[int, list[Concept]] = {i: [] for i in range(len(original_positions))}

        with results_file.open("r") as f:
            for line in f:
                result = json.loads(line)
                custom_id = result["custom_id"]

                # Check for errors
                if result.get("error"):
                    failed_requests.append(
                        {
                            "custom_id": custom_id,
                            "error": result["error"],
                        }
                    )
                    continue

                # Extract mapping info
                mapping_info = mapping[custom_id]
                pos_idx = mapping_info["position_idx"]
                concept_name = mapping_info["concept_name"]

                # Parse the response body
                response_body = result["response"]["body"]

                # Extract validation from response output
                validation = self._extract_validation(response_body)

                if validation is None:
                    failed_requests.append(
                        {
                            "custom_id": custom_id,
                            "error": "Failed to parse validation from response",
                        }
                    )
                    continue

                # Create refined concept
                if validation["is_valid"] and validation["temporal"] is not None:
                    refined_concept = Concept(
                        name=concept_name,
                        validated_by="llm",
                        temporal=validation["temporal"],
                        reasoning=validation["reasoning"],
                    )
                else:
                    refined_concept = Concept(
                        name=concept_name,
                        validated_by=None,
                        temporal=None,
                        reasoning=validation["reasoning"],
                    )

                new_concepts_by_position[pos_idx].append(refined_concept)

        # Update positions with refined concepts
        for pos_idx, concepts in new_concepts_by_position.items():
            original_positions[pos_idx].concepts = concepts

        # Log failed requests if any
        if failed_requests:
            batch_id = results_file.stem.replace("batch_output_", "")
            error_file = self.batch_dir / f"errors_{batch_id}.json"
            with error_file.open("w") as f:
                json.dump(failed_requests, f, indent=2)

        return original_positions

    def _find_mapping_file(self, results_file: Path) -> Path:
        """Find the mapping file corresponding to a results file.

        Args:
            results_file: Path to batch results file

        Returns:
            Path to corresponding mapping file
        """
        # Extract timestamp from results file name
        # results_file format: batch_output_{batch_id}.jsonl
        # We need to find batch_mapping_{timestamp}.json for the same batch

        # Find the batch metadata file
        batch_id = results_file.stem.replace("batch_output_", "")
        metadata_file = self.batch_dir / f"batch_{batch_id}.json"

        if not metadata_file.exists():
            raise ValueError(f"Cannot find metadata file for batch {batch_id}")

        with metadata_file.open("r") as f:
            metadata = json.load(f)

        # Extract timestamp from batch_file path
        batch_file_path = Path(metadata["batch_file"])
        timestamp = batch_file_path.stem.replace("batch_input_", "")

        mapping_file = self.batch_dir / f"batch_mapping_{timestamp}.json"
        if not mapping_file.exists():
            raise ValueError(f"Cannot find mapping file: {mapping_file}")

        return mapping_file

    def _extract_validation(self, response_body: dict[str, Any]) -> dict[str, Any] | None:
        """Extract ConceptValidation from OpenAI response body.

        Args:
            response_body: Response body from batch API result

        Returns:
            Dictionary with is_valid, temporal, and reasoning fields, or None if parsing fails
        """
        try:
            # Navigate through response structure
            output = response_body.get("output", [])

            # Find message item
            message = None
            for item in output:
                if item.get("type") == "message":
                    message = item
                    break

            if not message:
                return None

            # Extract content
            content = message.get("content", [])
            if not content:
                return None

            # Get parsed output
            text_item = content[0]
            if text_item.get("type") != "output_text":
                return None

            parsed = text_item.get("parsed")
            if not parsed:
                return None

            return {
                "is_valid": parsed["is_valid"],
                "temporal": parsed.get("temporal"),
                "reasoning": parsed["reasoning"],
            }
        except (KeyError, IndexError, TypeError):
            return None

    def wait_and_process(
        self,
        batch_id: str,
        original_positions: list[LabelledPosition],
        poll_interval: int = 60,
        timeout: int | None = None,
    ) -> list[LabelledPosition]:
        """Wait for batch completion and process results.

        Args:
            batch_id: Batch job ID to monitor
            original_positions: Original positions list to update
            poll_interval: Seconds to wait between status checks (default: 60)
            timeout: Maximum seconds to wait before raising TimeoutError (default: None for unlimited)

        Returns:
            Updated list of positions with validated concepts

        Raises:
            TimeoutError: If batch doesn't complete within timeout
            RuntimeError: If batch fails or is cancelled
        """
        start_time = time.time()

        while True:
            status_info = self.poll_status(batch_id)
            status = status_info["status"]

            if status == "completed":
                results_file = self.download_results(batch_id)
                return self.process_results(results_file, original_positions)
            elif status in ["failed", "expired", "cancelled"]:
                raise RuntimeError(f"Batch {batch_id} {status}")

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Batch {batch_id} did not complete within {timeout} seconds")

            time.sleep(poll_interval)

    def get_batch_metadata(self, batch_id: str) -> dict[str, Any]:
        """Retrieve saved metadata for a batch job.

        Args:
            batch_id: Batch job ID

        Returns:
            Dictionary with batch metadata

        Raises:
            ValueError: If metadata file doesn't exist
        """
        metadata_file = self.batch_dir / f"batch_{batch_id}.json"
        if not metadata_file.exists():
            raise ValueError(f"No metadata found for batch {batch_id}")

        with metadata_file.open("r") as f:
            return json.load(f)
