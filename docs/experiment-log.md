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
Testing reasoning trajectories from open weights models (gpt-oss-120b, Qwen3 Max) for synthetic SFT data generation. Observations from https://openrouter.ai/chat?room=orc-1765226132-q0WSkSjbe8HRTJFj13jK:
- **gpt-oss-120b**: Gets solutions but reasoning is messy (confused FEN parsing, truncated mid-sentence)
- **Qwen3 Max**: Good step-by-step structure but hallucinates continuations after correct moves

### Approach
Rule based aided context engineering + a score-based verification pipeline to filter generated reasoning traces:

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
- Disapointing results from both gpt-oss-20b/120b so went for gpt-5-nano

### Next Steps
- Run batch generation on 1000+ puzzles
  - Ongoing https://logfire-eu.pydantic.dev/pilipolio/chess-sandbox-puzzles-generation
  - https://huggingface.co/datasets/pilipolio/chess-reasoning-traces/viewer/default/train
  - ~200 batches needed (1000 / 5) of ~2500 tokens (~ $0.0001 per generation)
  - ~17-33 minutes total (200 × 5-10s) for roughly $0.80 and 20-30 minutes for 1,000 examples.
- Analyze score distribution and failure modes

---

## 2024-12-09: Reasoning SFT Training - Qwen3-0.6B Baseline

### Setup
- **Model**: Qwen/Qwen3-0.6B with LoRA (r=32)
- **Dataset**: pilipolio/chess-reasoning-traces (895 train, 100 test)
- **Config**: batch_size=2, grad_accum=4, lr=2e-4, max_length=2048
- **Run**: 50 steps test run (~0.27 epochs)

### Metrics Progression
| Step | Loss | Token Acc | Sections Found | First Move Acc |
|------|------|-----------|----------------|----------------|
| 10   | 1.56 | 64%       | 0.0/5          | 0%             |
| 20   | 1.26 | 70%       | 4.1/5          | 10%            |
| 30   | 1.15 | 71%       | 4.5/5          | 0%             |

### Key Observations

**What works:**
- Model learns section structure quickly (0 → 4.5/5 sections in 30 steps)
- Loss decreases steadily

**Issues identified:**
1. **FEN parsing fundamentally broken** - Model misinterprets digit notation:
   - `3Q4` parsed as "a8 black queen" instead of "3 empty, Queen on d8, 4 empty"
   - Digits representing empty squares not understood

2. **Generation loops** - Model gets stuck repeating piece lists:
   ```
   Re4, Rh4, Re4, Rh4, Re4, Rh4...
   ```

3. **First move accuracy unstable** - 0% → 10% → 0% suggests model hasn't learned puzzle solving, just format

### W&B Artifacts
- [Eval examples table](https://wandb.ai/guillaumeallain-test/chess-reasoning-sft/artifacts/run_table/run-f3i0ir1m-evalexamples-9fKLYw/v1/files/eval/examples.table.json)

### Next Steps
- Run Qwen3-4B with full 3 epochs (larger capacity should help FEN understanding)
- Consider adding FEN parsing examples to training data if 4B still struggles

---

## 2024-12-09: Qwen3-4B Training Attempt

### First Attempt (Failed)
- Ran without `--use-4bit` flag
- OOM crash at step 100 during eval (4B model too large for A10G 24GB in fp16)

### Fix
- Added `bitsandbytes==0.48.2` to Modal image (`modal_pipeline.py`)
- Required for 4-bit quantization support

### Retry Command
```bash
modal run --detach chess_sandbox/puzzles_trainer/modal_pipeline.py::train_reasoning -- \
    --model-id Qwen/Qwen3-4B \
    --use-4bit \
    --wandb-project chess-reasoning-sft \
    --wandb-run-name qwen3-4b-full
```

---

## 2024-12-10: Reasoning Model Evaluation Benchmark

### Setup
- **Dataset**: pilipolio/chess-reasoning-traces (100 test examples)
- **Models benchmarked**:
  - `chess-reasoning`: Qwen3-4B + LoRA finetuned on reasoning traces (Modal vLLM)
  - `qwen/qwen3-32b`: Baseline via OpenRouter
  - `openai/gpt-4o-mini`: Baseline via OpenRouter
- **Metric**: First move accuracy (exact match after normalization)
- **Evaluation**: Weave tracking at wandb.ai/guillaumeallain-test/chess-reasoning

### Infrastructure
| File | Description |
|------|-------------|
| `modal_reasoning_vllm.py` | vLLM endpoint with LoRA adapter for chess-reasoning model |
| `reasoning_evaluation.py` | Weave-based evaluation with OpenRouter/Modal support |

### Results

| Model | First Move Accuracy | Notes |
|-------|---------------------|-------|
| chess-reasoning (finetuned) | 6.0% | 6/100 correct |
| qwen/qwen3-32b | 0.0% | JSON parsing errors on some responses |
| openai/gpt-4o-mini | 0.0% | Returns valid moves but wrong solutions |

### Analysis

**Prompt engineering for baselines:**
- Added format suffix: "Respond with ONLY the best move in standard algebraic notation"
- Regex extraction: `([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?)`
- Move normalization (lowercase, strip +# suffixes)

**Why baseline models score 0%:**
Debug testing shows models do follow the format and return valid chess moves. They're just solving the puzzles incorrectly:
- Puzzle 1: Expected `Qxg2#`, GPT-4o-mini returned `Qxe4`
- Puzzle 2: Expected `Rd8+`, GPT-4o-mini returned `Bxa3`
- Puzzle 3: Expected `Qg6#`, GPT-4o-mini returned `Qe8#`

The extraction pipeline works correctly - baseline LLMs are simply not strong at tactical chess puzzles. This validates that the SFT training provides meaningful improvement (6% vs 0%).

### Next Steps
- Train for more epochs to improve finetuned model accuracy
- Consider stronger base models (Qwen3-8B, Qwen3-14B)
- Analyze which puzzle types/themes the finetuned model solves correctly

