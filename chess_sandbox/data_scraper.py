#!/usr/bin/env python3
"""
Chess Blog HTML Scraper - Extract structured chess position data
"""

import json
import os
from pathlib import Path
from typing import List

from openai import OpenAI
from pydantic import BaseModel, Field


class ThemedChessPosition(BaseModel):
    themes: List[str] = Field(
        description="an array of chess themes and motifs found in the text paragraph just before the actual position"
    )
    fen: str = Field(description="the entire FEN string from the position")


class ChessPositions(BaseModel):
    positions: List[ThemedChessPosition]


def scrape_html_file(html_path: str, output_path: str, model: str = "gpt-4o-mini") -> None:
    """Scrape HTML file and extract chess positions."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    with open(html_path, "r") as f:
        html_content = f.read()

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a chess expert extracting structured data from chess blog HTML."},
            {
                "role": "user",
                "content": f"""Scrape this html source page of a chess learning blog to extract \
all chess positions and the corresponding themes.

HTML content:
{html_content}""",
            },
        ],
        response_format=ChessPositions,
    )

    result = completion.choices[0].message.parsed
    if not result:
        raise ValueError("Failed to parse chess positions from HTML")

    with open(output_path, "a") as f:
        for position in result.positions:
            if position.fen in html_content:
                f.write(json.dumps(position.model_dump()) + "\n")
            else:
                print(f"WARNING: FEN not found in HTML: {position.fen}")


def main() -> None:
    """Main function to process all HTML files."""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    processed_dir.mkdir(parents=True, exist_ok=True)

    for html_file in raw_dir.glob("*.html"):
        output_file = processed_dir / f"{html_file.stem}.jsonl"
        print(f"Processing {html_file.name}...")
        scrape_html_file(str(html_file), str(output_file))
        print(f"Written to {output_file}")


if __name__ == "__main__":
    main()
