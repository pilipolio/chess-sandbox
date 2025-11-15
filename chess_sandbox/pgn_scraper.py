"""PGN scraping pipeline for downloading and processing annotated chess games."""

import json
import re
import zipfile
from pathlib import Path

import click
import httpx
from huggingface_hub import HfApi
from tqdm import tqdm

from chess_sandbox.concept_extraction.labelling.labeller import LabelledPosition
from chess_sandbox.concept_extraction.labelling.parser import extract_positions, parse_pgn_file


def extract_pgn_links(html: str, base_url: str) -> list[dict[str, str]]:
    """Extract PGN download links from HTML content.

    Args:
        html: HTML content to parse
        base_url: Base URL for resolving relative links

    Returns:
        List of dicts with 'url' and 'title' keys
    """
    links: list[dict[str, str]] = []

    # Match links ending in .pgn or .zip
    pattern = r'href=["\']([^"\']*\.(?:pgn|zip))["\']'
    matches = re.finditer(pattern, html, re.IGNORECASE)

    for match in matches:
        url = match.group(1)

        # Make absolute URL if relative
        if not url.startswith(("http://", "https://")):
            if url.startswith("/"):
                # Absolute path
                base = base_url.rstrip("/")
                url = f"{base}{url}"
            else:
                # Relative path
                base = base_url.rsplit("/", 1)[0]
                url = f"{base}/{url}"

        # Try to find title in <a> tag text or nearby headings
        title_match = re.search(r">([^<]+)</a>", html[match.start() : match.end() + 100])
        if title_match:
            title = title_match.group(1).strip()
        else:
            # Use filename as fallback
            title = url.split("/")[-1]

        links.append({"url": url, "title": title})

    return links


def download_file(url: str, output_dir: Path) -> Path | None:
    """Download a file from URL.

    Args:
        url: URL to download from
        output_dir: Directory to save file

    Returns:
        Path to downloaded file, or None if failed
    """
    try:
        filename = url.split("/")[-1]
        # Clean filename
        filename = re.sub(r"[^\w\-_\. ]", "_", filename)
        output_path = output_dir / filename

        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()

            output_path.write_bytes(response.content)
            return output_path

    except Exception as e:
        click.echo(f"Error downloading {url}: {e}", err=True)
        return None


def extract_zip_file(zip_path: Path, extract_dir: Path) -> list[Path]:
    """Extract PGN files from a ZIP archive.

    Args:
        zip_path: Path to ZIP file
        extract_dir: Directory to extract to

    Returns:
        List of extracted PGN file paths
    """
    pgn_files: list[Path] = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Extract all files
            zip_ref.extractall(extract_dir)

            # Find all PGN files in the extracted content
            for extracted_file in extract_dir.rglob("*"):
                if extracted_file.is_file() and extracted_file.suffix.lower() == ".pgn":
                    pgn_files.append(extracted_file)

    except Exception as e:
        click.echo(f"Error extracting {zip_path}: {e}", err=True)

    return pgn_files


def process_pgn_file(pgn_path: Path) -> list[LabelledPosition]:
    """Process a PGN file and extract annotated positions.

    Args:
        pgn_path: Path to PGN file

    Returns:
        List of labeled positions with comments
    """
    positions: list[LabelledPosition] = []

    try:
        games = parse_pgn_file(pgn_path)

        for i, game in enumerate(games):
            game_id = f"{pgn_path.stem}_game_{i}"
            game_positions = extract_positions(game, game_id)
            positions.extend(game_positions)

    except Exception as e:
        click.echo(f"Error parsing {pgn_path}: {e}", err=True)

    return positions


def save_positions_to_jsonl(positions: list[LabelledPosition], output_path: Path) -> None:
    """Save positions to JSONL file.

    Args:
        positions: List of labeled positions
        output_path: Path to output JSONL file
    """
    with output_path.open("w") as f:
        for position in positions:
            data = {
                "fen": position.fen,
                "comment": position.comment,
                "game_id": position.game_id,
                "move_number": position.move_number,
                "side_to_move": position.side_to_move,
                "move_san": position.move_san,
                "previous_fen": position.previous_fen,
                "concepts": [c.to_dict() for c in position.concepts] if position.concepts else [],
            }
            f.write(json.dumps(data) + "\n")


def upload_to_hf(
    file_path: Path,
    repo_id: str,
    filename: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> str:
    """Upload a file to HuggingFace dataset repository.

    Args:
        file_path: Local file to upload
        repo_id: HuggingFace dataset repository ID
        filename: Filename in the repo (defaults to file_path.name)
        revision: Git revision/branch to upload to
        token: HF token (if not set, uses HF_TOKEN env var)

    Returns:
        URL of uploaded file
    """
    api = HfApi(token=token)

    if filename is None:
        filename = file_path.name

    result = api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        commit_message=f"Add {filename} from PGN scrape",
    )

    return result  # type: ignore[no-any-return]


@click.command()
@click.option(
    "--url",
    type=str,
    default="https://www.angelfire.com/games3/smartbridge/",
    help="URL to scrape PGN files from",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="data/pgn_downloads",
    help="Directory to save downloaded PGN files",
)
@click.option(
    "--output-jsonl",
    type=click.Path(path_type=Path),
    default="data/scraped_positions.jsonl",
    help="Output JSONL file for extracted positions",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of PGN files to download (for testing)",
)
@click.option(
    "--upload-to-hub",
    is_flag=True,
    default=False,
    help="Upload results to HuggingFace dataset",
)
@click.option(
    "--hf-repo-id",
    type=str,
    default=None,
    help="HuggingFace dataset repository ID (required if --upload-to-hub)",
)
@click.option(
    "--hf-revision",
    type=str,
    default=None,
    help="HuggingFace dataset revision/branch",
)
@click.option(
    "--hf-token",
    type=str,
    default=None,
    help="HuggingFace API token (uses HF_TOKEN env var if not provided)",
)
def main(
    url: str,
    output_dir: Path,
    output_jsonl: Path,
    limit: int | None,
    upload_to_hub: bool,
    hf_repo_id: str | None,
    hf_revision: str | None,
    hf_token: str | None,
) -> None:
    """Scrape PGN files from a webpage and extract annotated positions.

    This pipeline:
    1. Scrapes a webpage for PGN download links
    2. Downloads the PGN files
    3. Extracts positions with comments/annotations
    4. Saves to JSONL format
    5. Optionally uploads to HuggingFace dataset
    """
    if upload_to_hub and not hf_repo_id:
        raise click.UsageError("--hf-repo-id is required when --upload-to-hub is set")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Scrape webpage for PGN links
    click.echo(f"Scraping {url} for PGN files...")
    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
            html = response.text
    except Exception as e:
        click.echo(f"Error fetching URL: {e}", err=True)
        return

    links = extract_pgn_links(html, url)

    if not links:
        click.echo("No PGN or ZIP files found on the page.", err=True)
        return

    click.echo(f"Found {len(links)} files (PGN and ZIP)")

    if limit:
        links = links[:limit]
        click.echo(f"Limited to first {limit} files")

    # Step 2: Download files (PGN and ZIP)
    click.echo("Downloading files...")
    downloaded_files: list[Path] = []
    for link in tqdm(links, desc="Downloading"):
        file_path = download_file(link["url"], output_dir)
        if file_path:
            downloaded_files.append(file_path)

    click.echo(f"Successfully downloaded {len(downloaded_files)} files")

    # Step 3: Extract ZIP files and collect all PGN files
    click.echo("Processing files and extracting ZIPs...")
    pgn_files: list[Path] = []
    extract_dir = output_dir / "extracted"

    for file_path in tqdm(downloaded_files, desc="Processing"):
        if file_path.suffix.lower() == ".zip":
            # Extract ZIP and add extracted PGN files
            extracted_dir = extract_dir / file_path.stem
            extracted_dir.mkdir(parents=True, exist_ok=True)
            extracted_pgns = extract_zip_file(file_path, extracted_dir)
            pgn_files.extend(extracted_pgns)
        elif file_path.suffix.lower() == ".pgn":
            # Direct PGN file
            pgn_files.append(file_path)

    zip_count = len([f for f in downloaded_files if f.suffix.lower() == ".zip"])
    click.echo(f"Found {len(pgn_files)} PGN files (including {zip_count} from ZIPs)")

    # Step 4: Extract annotated positions from all PGN files
    click.echo("Extracting annotated positions...")
    all_positions: list[LabelledPosition] = []
    for pgn_file in tqdm(pgn_files, desc="Extracting positions"):
        positions = process_pgn_file(pgn_file)
        all_positions.extend(positions)

    click.echo(f"Extracted {len(all_positions)} annotated positions")

    # Step 5: Save to JSONL
    if all_positions:
        save_positions_to_jsonl(all_positions, output_jsonl)
        click.echo(f"Saved positions to {output_jsonl}")
    else:
        click.echo("No annotated positions found in downloaded files.", err=True)
        return

    # Step 6: Upload to HuggingFace (optional)
    if upload_to_hub:
        click.echo(f"Uploading to HuggingFace dataset {hf_repo_id}...")
        try:
            result_url = upload_to_hf(
                output_jsonl,
                hf_repo_id,  # type: ignore[arg-type]
                revision=hf_revision,
                token=hf_token,
            )
            click.echo(f"Successfully uploaded to: {result_url}")
        except Exception as e:
            click.echo(f"Error uploading to HuggingFace: {e}", err=True)


if __name__ == "__main__":
    main()
