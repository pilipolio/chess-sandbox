---
name: chess-commentator
description: This skill should be used when analyzing chess positions. Automatically triggers when users provide FEN positions for analysis or ask about specific chess positions. Provides engine-powered analysis with natural language explanations of best moves, key variations, and strategic/tactical themes.
---

# Chess Commentator

## Overview

Analyze chess positions using Stockfish engine analysis combined with natural language explanations. Provide succinct commentary focusing on the best move, why it's best, and key variations with thematic insights.

## When to Use This Skill

This skill automatically triggers when:
- User provides a FEN string to analyze
- User asks to analyze a chess position
- User asks about best moves in a specific position
- User requests position evaluation or commentary

## Core Workflow

### 1. Analyzing a Position

To analyze a chess position from a FEN string:

```bash
uv run python -m chess_sandbox.engine.analysis "<FEN>"
```

Example:
```bash
uv run python -m chess_sandbox.engine.analysis "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
```

### 2. Analyzing After a Specific Move

To analyze a position after a specific move is played:

```bash
uv run python -m chess_sandbox.engine.analysis "<FEN>" --next-move <MOVE_IN_SAN>
```

Example:
```bash
uv run python -m chess_sandbox.engine.analysis "8/8/2K5/p1p5/P1P5/1k6/8/8 w - - 0 58" --next-move Kb5
```

**Important:** The move must be in Standard Algebraic Notation (SAN), e.g., "Nf3", "e4", "O-O", "Bxe5"

### 3. Extracting Concepts from a Position

To extract AI-detected chess concepts with confidence scores:

```bash
uv run python -m chess_sandbox.concept_extraction.model.inference predict "<FEN>"
```

Example:
```bash
uv run python -m chess_sandbox.concept_extraction.model.inference predict "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
```

**Output:** Concepts sorted by confidence descending, showing all concepts above 10% confidence (default threshold).

## Interpreting Engine Output

The engine_analysis module outputs:
1. **Position diagram** - Visual representation of the board
2. **FEN** - Position in FEN notation
3. **Turn** - Who is to move
4. **Top engine lines** - Multiple candidate moves with evaluations

Each line includes:
- **Depth** - How deeply the engine calculated
- **Eval** - Position evaluation in pawns (+positive = White better, -negative = Black better)
- **Moves** - The sequence of moves in SAN notation

## Providing Commentary

When analyzing positions, provide **succinct commentary** that includes:

### 0. ASCII representation of the position
As returned by the tool 

### 1. Position Assessment
Briefly state the evaluation (White better / Black better / Equal / Winning / Losing)

### 2. Best Move Explanation
- Identify the top engine move
- Explain **why** it's best in 1-2 sentences
- Focus on the move's purpose and what it accomplishes

### 3. Key Variations
- Present the main line(s) with brief annotations
- Highlight critical moments or decision points
- Use chess notation with explanatory comments

### 4. Thematic Insights
Identify relevant tactical or strategic themes present in the position:
- Tactical motifs (forks, pins, discoveries, etc.)
- Strategic concepts (weak squares, pawn structure, piece activity, etc.)
- Critical evaluation factors (king safety, material imbalance, initiative, etc.)

**Reference:** Load `references/chess_themes.md` when needed for comprehensive theme identification.

### 5. Detected Concepts (Optional)
When concept extraction is used, present AI-detected concepts after thematic insights:
- List concepts with confidence ≥ 10%
- Sort by confidence (highest first)
- Format: "Concept name (XX.X%)"
- Keep this section concise - typically 3-5 top concepts

## Commentary Style Guidelines

**DO:**
- Be succinct and focused
- Explain the "why" behind moves
- Use proper chess notation
- Identify concrete themes and patterns
- Compare alternatives when relevant

**DON'T:**
- Write verbose or overly long explanations
- State obvious information without insight
- Ignore the engine's top recommendations without good reason
- Provide generic advice without position-specific analysis

## Example Analysis Format

```
Position Assessment: White has a slight advantage (+0.35)

Best Move: Nf3
This develops the knight to its best square, controlling the center (e5, d4) while preparing kingside castling. It's more flexible than Nc3 as it doesn't block the c-pawn.

Key Variation:
1. Nf3 Nc6 2. d4 d5 3. c4 (Attacking the center, transitioning into a Queen's Gambit structure) 3... e6 4. Nc3 Nf6 (Symmetrical development, both sides complete development before committing to pawn breaks)

Themes:
- Central control: Both moves fight for central squares
- Development: Prioritizing piece activity before committing pawns
- Flexibility: Nf3 maintains options for c4 or e4 pawn breaks

Detected Concepts:
- Development advantage (87.3%)
- Central control (72.1%)
- King safety preparation (45.6%)
- Piece activity (23.4%)
```

## Additional Options

The engine_analysis module accepts optional parameters:
- `--depth <NUMBER>` - Analysis depth (default: 20)
- `--num-lines <NUMBER>` - Number of candidate moves to analyze (default: 5)
- `--stockfish-path <PATH>` - Custom Stockfish binary location

Example:
```bash
uv run python -m chess_sandbox.engine.analysis "<FEN>" --depth 25 --num-lines 3
```

## Resources

### references/chess_themes.md
Comprehensive reference of tactical and strategic themes. Load this file when detailed theme identification is needed or when encountering unfamiliar patterns.
