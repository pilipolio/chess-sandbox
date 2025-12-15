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

### Observations

**Model over-indexes on mate patterns:**
The finetuned model hallucinates mating ideas even on non-mate puzzles. Example on [skewer puzzle](https://lichess.org/training/029B5) (solution: `Rd8+` winning the queen):

```
## Step 4: Candidate Moves
Rxe5# captures the Black king on e5 with the rook from e8, delivering mate...
Rxe5#, Rxg6, Rxh6
```

The model fabricates a checkmate (`Rxe5#`) that doesn't exist, missing the actual skewer tactic entirely. This suggests:
1. Training data is mate-heavy (puzzles often end in checkmate)
2. Model learned "find mate" as a default heuristic rather than evaluating the actual position

**Dataset theme imbalance confirmed:**
- ~60-70% of puzzles are mateIn1/mateIn2
- Only ~5-8% fork, ~3-5% skewer, rare pins
- Model overfits to mate patterns, struggles with material-winning tactics

### Next Steps: GRPO with Verifiable Rewards

Move to reinforcement learning (GRPO) using verifiable chess rewards:
- **Legality reward**: -1.0 for illegal moves (hard constraint)
- **Engine eval**: Stockfish centipawn score normalized to [-1, 1]
- **Format bonus**: Small reward for valid `<think>` structure

### Open Questions for GRPO

1. **Relaxed thinking vs structured output**: Should we keep the strict 5-step format or let the model reason freely? Free-form may discover better patterns but loses interpretability.

2. **Rewarding intermediate steps**: Should we verify intermediate reasoning (piece positions, candidate moves) or only reward the final move? Options:
   - Final move only (simpler, lets model discover its own reasoning style)
   - Verify piece positions accuracy (grounds the model in board state)
   - Verify candidate moves are legal (catches hallucinations early)

3. **Structured output for grounding**: Force model to output piece positions and candidate moves as structured data before the solution? This could:
   - Prevent hallucinations like the fake `Rxe5#`
   - Add computational overhead
   - Be verified programmatically as part of reward

---

## 2024-12-11: Model Comparison - gpt-4.1-nano vs gpt-5-nano

### Motivation
Eyeballing 3 samples from gpt-4.1-nano revealed systematic issues with reasoning quality despite correct final answers. Tested gpt-5-nano as alternative.

### Results

| Metric | gpt-4.1-nano | gpt-5-nano |
|--------|--------------|------------|
| Verification score | 0.84-0.93 | **1.00** (3/3) |
| First move correct | 3/3 | 3/3 |
| FEN parsing | Broken (phantom pieces) | **Accurate** (per-square) |
| Piece positions | Hallucinated | **Correct** |
| Solution in candidates | 1/3 | **3/3** |
| Reasoning supports answer | 0/3 | **3/3** |

### Key Issues with gpt-4.1-nano
1. **FEN parsing broken**: Digits misinterpreted (e.g., `5rk1` → phantom h8 rook)
2. **Piece positions hallucinated**: Lists pawns on non-existent squares
3. **Post-hoc rationalization**: Explores wrong moves, then outputs correct solution
4. **Candidate moves miss solution**: Often doesn't include the winning move

### gpt-5-nano Improvements
- Square-by-square FEN breakdown: `f8 black rook, g8 black king, h8 empty`
- Solution move always in candidate list with explanation
- Refuted lines shown: explains WHY alternatives fail
- Reasoning genuinely supports conclusion

### Decision
**Default model changed to `openai/gpt-5-nano`** for reasoning trace generation.

For larger batch runs, consider `openai/gpt-5-mini` for higher quality at ~3x cost.

### Recommended Models (OpenRouter)
| Use Case | Model | Notes |
|----------|-------|-------|
| Quick iteration | `openai/gpt-5-nano` | Fast, cheap, good quality |
| Production SFT data | `openai/gpt-5-mini` | Higher quality reasoning |
| Baseline comparison | `openai/gpt-4o-mini` | Alternative provider |

---

## 2024-12-11: GRPO Pipeline - First Test Runs

### Setup
- **Implementation**: TRL 0.25.1 GRPOTrainer with custom chess reward function
- **Infrastructure**: Modal A10G (24GB), 8hr timeout
- **Reward function**: 4-component weighted score
  - Legality (40%): -1.0 for illegal first move
  - Correctness (40%): First move matches puzzle solution
  - Format (15%): 5 reasoning sections present
  - Piece accuracy (5%): Board awareness from Step 2

### Files Created
| File | Description |
|------|-------------|
| `grpo_rewards.py` | Reward functions reusing reasoning_verifier |
| `grpo_trainer.py` | GRPOTrainer wrapper with Click CLI |
| `modal_grpo.py` | Modal deployment config |

### Test Results

| Model | Max Steps | Reward Mean | Clipped Ratio | Issue |
|-------|-----------|-------------|---------------|-------|
| Qwen/Qwen3-0.6B | 10 | -1.0 | 100% | Base model doesn't know format |
| pilipolio/chess-reasoning-sft-qwen3-0.6b | 10 | -1.0 | 100% | Completions truncated |
| pilipolio/chess-reasoning-sft-qwen3-4b | 10 | -1.0 | 100% | Completions truncated |

### Root Cause Analysis

**Problem**: All completions hit max length (512 tokens) without terminating.

**Evidence**:
- `completions/clipped_ratio: 1.0` (100% clipped)
- `completions/mean_terminated_length: 0.0` (no EOS tokens)
- `reward: -1.0` (all marked illegal due to truncated output)

**Dataset analysis**:
```
Answer lengths (chars):
  Mean: 1705
  95th percentile: 2317
  Approx tokens: 426 avg, 579 95th pct
```

The `max_completion_length=512` is too short - 95th percentile needs ~580 tokens.

### Potential Issues

1. **Max length too short**: Need 768-1024 tokens for full reasoning traces
2. **Model not terminating**: Even with more tokens, model may not emit EOS
3. **LoRA stacking**: Warning about "multiple adapters" when loading SFT checkpoint

### Next Steps

1. **Increase max_completion_length** to 1024 tokens
2. **Debug generation**: Sample a few completions to verify format
3. **Check EOS behavior**: Verify tokenizer pad/eos tokens match training
4. **Consider curriculum**: Start with shorter puzzles or legality-only reward

---

## 2024-12-12: Debug Tools & Model Comparison

### Motivation
GRPO pipeline showing -1.0 rewards due to truncation. Added debug tooling to diagnose generation behavior before fixing.

### Implementation

**New files:**
| File | Description |
|------|-------------|
| `grpo_debug.py` | Sample generations with reward diagnostics |
| `test_eos_behavior.py` | Test EOS token termination behavior |
| `modal_grpo_debug.py` | Modal deployment for GPU-accelerated debug |

**Fix applied:**
- `grpo_trainer.py`: `max_completion_length` default changed from 512 to 1024

**CLI entry points added:**
- `uv run grpo-debug --model-id MODEL --num-samples 5 --verbose`
- `uv run test-eos --base-model Qwen/Qwen3-0.6B`

### Results: 0.6B SFT Model (Local)

| Metric | Value |
|--------|-------|
| EOS emitted | 0/3 |
| Has `</think>` | 0/3 |
| Correct move | 0/3 |
| Average reward | -1.0 |

**Issue:** Model stuck in degenerate generation loops:
```
Kf1# (no mate available)
Kf1# (no mate available)
Kf1# (no mate available)
...
```

### Results: 4B SFT Model (Modal A10G)

| Metric | Value |
|--------|-------|
| EOS emitted | 4/5 |
| Has `</think>` | 4/5 |
| Correct move | 1/5 (20%) |
| Average length | ~800 tokens |

**Per-sample breakdown:**
| Sample | Expected | Got | Status |
|--------|----------|-----|--------|
| 0 | h6 | h6 | ✓ |
| 1 | Qf7# | None | ✗ (truncated at 1024) |
| 2 | Qxh2# | Qxg2# | ✗ (wrong square) |
| 3 | Qf1# | Qxh2+ | ✗ (wrong move) |
| 4 | Qxh2# | Qh1# | ✗ (wrong square) |

### Analysis

**0.6B model:**
- FEN parsing fundamentally broken (same issue as SFT training)
- Generation loops prevent termination
- Not viable for GRPO without significant improvements

**4B model:**
- Terminates properly with EOS and `</think>` tags
- Produces structured output with valid format
- Wrong moves suggest positional understanding issues, not format issues
- Good candidate for GRPO training (can generate rewards > -1.0)

### Conclusions

1. **Truncation fix validated**: 1024 tokens sufficient for 4/5 samples
2. **4B model ready for GRPO**: Proper termination enables meaningful reward signal
3. **0.6B model not viable**: Generation loops and broken FEN parsing prevent learning
4. **Next step**: Run GRPO training with 4B SFT checkpoint

### Note on Debug Files
The `modal_grpo_debug.py` file is a temporary debug utility. Consider removing after GRPO training is stable.

---

## 2024-12-13: GRPO Pipeline Status Update

### Current State

| Model | Status | Notes |
|-------|--------|-------|
| Qwen3-4B SFT | Ready for GRPO | Proper termination, ~20% first move accuracy |
| Qwen3-0.6B SFT | Not viable | Generation loops, broken FEN parsing |

**Key findings from debug:**
- `max_completion_length=1024` sufficient for 4/5 samples
- 4B model emits EOS and `</think>` tags correctly
- Wrong moves indicate positional understanding gaps, not format issues

### Next Steps for GRPO Training

1. **Run full GRPO training with 4B SFT checkpoint**
   ```bash
   modal run chess_sandbox/puzzles_trainer/modal_grpo.py -- \
       --model-id pilipolio/chess-reasoning-sft-qwen3-4b \
       --num-generations 8 \
       --max-completion-length 1024 \
       --kl-coef 0.04
   ```

2. **Reward function refinement**
   - Current: 4-component (legality 40%, correctness 40%, format 15%, pieces 5%)
   - Consider: PGN validation of ALL moves in solution, not just first move
   - Consider: Engine eval of final position (continuous signal)

3. **Curriculum options to explore**
   - Start with higher-legality puzzles (mate-in-1, clear forcing moves)
   - Balanced theme sampling (use `THEME_QUOTAS` from `reasoning_generator.py`)

4. **Monitoring metrics**
   - `completions/clipped_ratio` < 100% (model terminating properly)
   - `reward/mean` improvement over baseline
   - `first_move_accuracy` on held-out test set

### Documentation Updates

Updated `docs/chess-llm-finetuning.md`:
- Added "Reasoning Trace Pipeline" section explaining:
  - PGN format as bridge to RLVR
  - Theme injection for domain grounding
  - 5-step reasoning structure
  - Contrastive learning via Maia + Stockfish refutations
- Condensed Experiment Log to summary table with pointer to this file
- Updated GRPO reward design to reference PGN validation
