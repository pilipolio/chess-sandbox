# Experiment Log

## 2024-12-07: Pydantic-AI Agent for Puzzle Reasoning

### Approach
Built a pydantic-ai agent (`puzzle_agent.py`) equipped with Stockfish analysis tools to generate chess teacher-style explanations for puzzles in PGN notation.

**Tools provided:**
- `evaluate_moves_tool`: Compare candidate moves using Stockfish evaluation

**Prompt strategy:**
- Provide puzzle context (FEN, theme, expected solution)
- Instruct agent to stop at the end of the expected solution
- Ask for alternatives at each move with short refutations

### Test Position
```
FEN: 5rk1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 b - - 2 27
Theme: Mating threat
Solution: 27...Rd8 28.Qxd8+ Bxd8
```

### Results

**What worked:**
- Agent correctly identifies the solution moves
- Stops at the end of the expected solution (no deep continuation)
- Shows Black's alternatives with refutations (e.g., 27...Qxd6? 28.Rxd6)

**What didn't work:**
- Agent doesn't explore White's alternatives at move 28
- The crux of the puzzle is: **why is 28.Qxd8+ forced?**
- Answer: 28.Qxb6?? Rxd1+ 29.Ne1 Rxe1# (back-rank mate)
- The agent only evaluates alternatives for the side to move in the solution, not the opponent's defensive tries

### Sample Output
```
27... Rd8 {seizes the d-file and creates an unstoppable threat}
  (27... Qxd6? {tempting capture} 28. Rxd6 {wins material})
28. Qxd8+ {forced}
28... Bxd8 {completing the tactic}
```

### Next Steps
- Consider adding a tool or prompt instruction to explore opponent's alternatives
- The agent needs to ask "what if White doesn't play Qxd8+?" to find the back-rank mate threat

---

## 2024-12-09: Reasoning Trace Verification for SFT Data

### Motivation
Testing reasoning trajectories from open weights models (gpt-oss-120b, Qwen3 Max) for synthetic SFT data generation. Observations:
- **gpt-oss-120b**: Gets solutions but reasoning is messy (confused FEN parsing, truncated mid-sentence)
- **Qwen3 Max**: Good step-by-step structure but hallucinates continuations after correct moves

### Approach
Built a score-based verification pipeline to filter generated reasoning traces:

1. **Updated prompt template** - Guide models to produce structured output:
   - Position Analysis (Qwen-style)
   - Tactical Assessment
   - Solution (PGN annotation style with `{curly bracket comments}`)

2. **Verifier module** (`reasoning_verifier.py`) with composite scoring:
   - Section completeness (30%): All three sections present
   - Move legality (40%): Validate moves in Solution section with python-chess
   - First move correctness (30%): Must match puzzle solution

3. **CLI integration** with filtering options:
   - `--min-score 0.6` (default threshold)
   - `--include-failed` for analysis

### Target Output Format
```
<think>
## Position Analysis
- Side to move: White
- Material: Equal
- Key features: Knight on g5 attacking f7/h7, Black king on g8 with weak back rank

## Tactical Assessment
- Candidate checks: Nxh7+, Qh7+
- Themes: Back-rank weakness, knight sacrifice

## Solution
19. Nxh7! {Deflection sacrifice - removes the h7 pawn}
19...Kxh7 {Forced}
20. Qh7# {Checkmate}
</think>
Nxh7
```

### Implementation
| File | Description |
|------|-------------|
| `reasoning_verifier.py` | Score-based validation with section parsing, PGN extraction, move legality |
| `reasoning_generator.py` | Updated prompt + verifier integration + CLI options |
| `test_reasoning_verifier.py` | 24 test cases |

### Results
- Verifier correctly identifies missing sections, illegal moves, wrong first moves
- Score-based filtering allows 1-2 illegal side-line moves while requiring correct solution
- Ready for gpt-oss-120b batch generation with quality filtering

### Next Steps
- Run batch generation on 1000+ puzzles
- Analyze score distribution and failure modes
- Iterate on prompt to improve section compliance

---

## 2024-12-09: Prompt Optimization for Section Compliance

### Problem
Initial runs with `gpt-oss-20b:free` showed poor format compliance:
- Verification score: avg=0.37
- Sections found: 0-1 out of 3
- Model spent tokens on FEN decoding instead of analysis

### Root Cause
Format instructions appeared at the END of the prompt. Model focused on position data and ignored structure requirements.

### Solution
Restructured `REASONING_PROMPT_TEMPLATE`:

1. **Format first**: Put output format at the TOP of prompt
2. **Compact layout**: Condensed position info to single lines
3. **Clear directive**: "Analyze this chess puzzle. Output EXACTLY:"
4. **Direct ending**: "Start with `<think>` and end with just the move {first_move}"

### Results

| Metric | Before | After |
|--------|--------|-------|
| Avg Score | 0.37 | **0.80** |
| Max Score | 0.40 | **1.00** |
| Sections Found | 0-1/3 | **3/3** |
| Illegal Moves | Example text copied | **None** |

### Observations
- Model still meta-reasons about format rather than doing chess analysis
- This is a limitation of `gpt-oss-20b:free`, not the prompt
- Better models (gpt-4o-mini, claude-haiku) should produce actual analysis
- Structure compliance now allows verification to work correctly

### Code Changes
- `reasoning_generator.py`: Simplified prompt template (~25 lines vs ~35 lines)
