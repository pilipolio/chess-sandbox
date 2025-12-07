# Fine-tuning Small LLMs for Chess

A two-stage training pipeline combining Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) to teach small language models to play chess.

## Approach Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Stage 1: SFT                        Stage 2: RL                           │
│   ───────────────                     ─────────────────                     │
│   Structured tasks with               RL with verifiable rewards            │
│   single correct answers              for more complex positions            │
│                                       with multiple acceptable lines        │ │                                                                             │
│   • Legal move detection              • Legality constraints                │
│   • Mate-in-1 puzzles                 • Reasoning verification (HF?)        │
│   • Forced sequences                  • Engine-scored rewards               │
│                                                                             │
│   Dataset: Lichess puzzles            Reward: Stockfish eval + legality     │
│                                       + reasoning preferences               │
│   Tool: TRL SFTTrainer                Tool: TRL GRPOTrainer                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Motivation:** Even strong reasoning models like GPT-5 lack chess board awareness and frequently produce illegal moves ([benchmark](https://blog.mathieuacher.com/GPT5-IllegalChessBench/)). We posit that LLMs, like humans, benefit from drilling patterns and board visualization ("System 1" intuition) before developing calculation and strategic understanding ("System 2" reasoning). The two-stage approach grounds the model in legal chess patterns through SFT, then refines look-ahead capabilities and strategic awareness through RL with verifiable rewards (GRPO).

## Stage 1: Supervised Fine-Tuning (SFT)

### Tasks

| Task | Input | Output | Purpose |
|------|-------|--------|---------|
| Board ↔ FEN | ASCII board diagram | FEN string (or vice versa) | Board visualization and spatial reasoning |
| Legal moves | FEN position | List of legal moves for a piece | Learn chess rules implicitly |
| Legal captures | FEN position | List of all legal captures | Identify tactical opportunities |
| Mate-in-1 | FEN + "Find checkmate" | Single move (SAN) | Tactical pattern recognition |
| Puzzle solving | FEN + puzzle prompt | Move sequence | Multi-step calculation |
| Best move | FEN + context | Single best move | Position evaluation |

### Datasets

**Primary: [Lichess/chess-puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles)** (5.52M puzzles)
- Rating-stratified difficulty
- Themes: mate, fork, pin, skewer, etc.
- Solution sequences provided

**Formatted: [pilipolio/lichess-puzzles-solutions](https://huggingface.co/datasets/pilipolio/lichess-puzzles-solutions)** (10k examples)
- Pre-formatted with question/answer pairs
- Fields: `fen`, `puzzle_solution`, `first_move`, `rating`, `themes`, `question`, `answer`
- Ready for instruction tuning

**Additional: [Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games)** (7.14B games)
- Full games for next-move prediction
- Can filter by rating for quality

### Implementation

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("pilipolio/lichess-puzzles-solutions")

def format_puzzle(example):
    return {
        "messages": [
            {"role": "user", "content": f"Position: {example['fen']}\n{example['question']}"},
            {"role": "assistant", "content": example["answer"]}
        ]
    }

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="./chess-sft", ...),
    train_dataset=dataset["train"].map(format_puzzle),
)
```

### Training Configuration

**GPU Memory Estimates (QLoRA)**

| Model | VRAM | Fits on |
|-------|------|---------|
| Qwen3-14B | ~14 GB | T4/A10G |
| Qwen3-4B | ~7 GB | Any GPU |
| Llama-3.1-8B | ~11 GB | T4/A10G |
| Llama-3.2-3B | ~5 GB | Any GPU |

*Source: [TRL SFT notebook](https://github.com/huggingface/trl/blob/794d87ff8d46fc41a07ffe9fb96771fee75922d4/examples/notebooks/sft_trl_lora_qlora.ipynb)*

**Recommended SFTConfig for A10G (24GB)**

```python
SFTConfig(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_length=512,
    packing=True,
    gradient_checkpointing=True,
    fp16=True,
)
```

**Memory-Saving Options**

For larger models or constrained GPUs:

```python
# 8-bit optimizer (reduces optimizer states ~75%)
optim="paged_adamw_8bit"

# QLoRA with double quantization
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
```

## Stage 2: Reinforcement Learning

### Why GRPO

[GRPO (Group Relative Policy Optimization)](https://arxiv.org/abs/2402.03300) from DeepSeek offers key advantages over PPO:

| Aspect | PPO | GRPO |
|--------|-----|------|
| Models required | Policy + Critic + Reward + Reference | Policy + Reference only |
| Memory overhead | ~4x model size | ~2x model size |
| Reward model | Neural network | Rule-based functions work |
| Complexity | High | Moderate |

**Core idea:** Sample multiple outputs per prompt, compute rewards, normalize across the group:

```
Advantage_i = (reward_i - mean(rewards)) / std(rewards)
```

This eliminates the need for a learned value function while providing stable training signal.

### Reward Design

Chess offers naturally verifiable rewards - a key advantage for GRPO:

```python
import chess
import chess.engine

def chess_reward(fen: str, predicted_move: str) -> float:
    board = chess.Board(fen)

    # Component 1: Legality (hard constraint)
    try:
        move = board.parse_san(predicted_move)
        if move not in board.legal_moves:
            return -1.0  # Illegal move penalty
    except ValueError:
        return -1.0  # Unparseable move

    # Component 2: Quality (engine evaluation)
    board.push(move)
    with chess.engine.SimpleEngine.popen_uci("stockfish") as engine:
        info = engine.analyse(board, chess.engine.Limit(depth=15))
        score = info["score"].relative

        # Normalize centipawn score to [-1, 1] range
        if score.is_mate():
            return 1.0 if score.mate() > 0 else -0.5
        cp = score.score()
        return max(-1.0, min(1.0, cp / 500))
```

**Reward components:**
1. **Legality** (-1.0 for illegal) - hard constraint
2. **Engine evaluation** - continuous signal from Stockfish
3. **Optional: Format compliance** - for CoT reasoning structure

### Implementation

```python
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    output_dir="./chess-grpo",
    learning_rate=1e-6,
    num_generations=8,  # Sample 8 moves per position
    max_new_tokens=64,
    kl_coef=0.04,  # Prevent drift from SFT checkpoint
)

def reward_function(prompts, completions):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        fen = extract_fen(prompt)
        move = extract_move(completion)
        rewards.append(chess_reward(fen, move))
    return rewards

trainer = GRPOTrainer(
    model=sft_model,  # Start from SFT checkpoint
    config=config,
    reward_funcs=[reward_function],
    train_dataset=positions_dataset,
)
```

## Model Candidates

| Model | Size | Notes |
|-------|------|-------|
| **[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)** | 21B (3.6B active) | OpenAI open-weights, already exposed to chess, Apache 2.0, fits in 16GB with MXFP4 |
| **gpt-4o-mini** (fine-tuning API) | ~20B? | Strong baseline via OpenAI API, already exposed to chess data |
| **Qwen3-3B+** | 3-8B | Good multilingual, strong reasoning |
| **[Ministral-3B-Reasoning](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512)** | 3B | Reasoning-optimized, also 8B+ variants available |

**Considerations:**
- gpt-oss-20b likely strongest starting point given pre-training exposure to chess
- ~3B params minimum for smaller open models
- Vision-capable models could use board images instead of/alongside FEN
- LoRA/QLoRA enables training larger models on consumer GPUs

## Representation Choices

### Position Encoding

**FEN (Forsyth-Edwards Notation)** - Primary choice
```
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
```
- Compact, unambiguous
- Standard in chess software
- Directly usable with python-chess
- Available for all puzzles

**Note on PGN move sequences:** For prompt engineering with large pre-trained models, providing the full game history as PGN (e.g., `1. e4 e5 2. Nf3 Nc6...`) can improve performance by leveraging patterns from pre-training data ([analysis](https://blog.mathieuacher.com/GPTsChessEloRatingLegalMoves/)). However, for fine-tuning smaller models, FEN is more token-efficient and puzzles typically lack game history context.

### Move Notation

**SAN (Standard Algebraic Notation)** - Selected
```
e4, Nf3, Bxc6+, O-O, Qxd8#
```
- Human-readable
- Matches training data (games, books, puzzles)
- Requires position context to parse

**UCI (Universal Chess Interface)**
```
e2e4, g1f3, b5c6, e1g1
```
- Unambiguous without position context
- Machine-friendly
- Less natural for LLMs

### Reasoning Format

**Experiment with both:**

Direct move:
```
User: Position: [FEN]. Find the best move.
Assistant: Nxe5
```

Chain-of-thought:
```
User: Position: [FEN]. Find the best move.
Assistant: <think>
The knight on c6 is pinned to the king. If I play Nxe5,
black cannot recapture due to the pin. This wins a pawn.
</think>
Nxe5
```

## Existing Infrastructure

The chess-sandbox codebase provides useful components:

- **`chess_sandbox/engine/analyse.py`** - Stockfish/Maia integration for evaluation
- **`chess_sandbox/engine/position_analysis.py`** - High-level position analysis
- **`chess_sandbox/concept_extraction/`** - Could provide additional reward signals (tactical concepts)

## Open Questions

### Training

1. **Curriculum strategy**: Should SFT start with simple (mate-in-1) → complex (mate-in-4+), or train on mixed difficulty?

2. **Legality pre-training**: Would a dedicated "legal move generator" phase before puzzles help?

3. **Game length in SFT**: ChessLLM found training on full games (vs. positions) improved Elo by ~350. Worth incorporating?

4. **GRPO hyperparameters**: Optimal `num_generations`, `kl_coef` for chess? DeepSeek used 64 samples - feasible with engine verification?

5. **Multi-stage vs mixed datasets**: Train sequentially (legal moves → puzzles → GRPO) or mix all SFT tasks together? Does staged curriculum prevent catastrophic forgetting or is interleaving more robust?

6. **Annotated reasoning data**: Use human-annotated puzzle explanations or synthetic CoT from GPT-5 as RL signal? Example format:
   ```
   Q: Solve puzzle FEN 1r4k1/4nppp/8/4Pb2/8/1P5P/r1PR4/3R3K w - - 0 27
   Long: 27. Rd8+ {Back-rank mate in two: force rook to d8 with check} Rxd8 {Forced} 28. Rxd8# {Mate}
   Short: 27. Rd8+ Rxd8 28. Rxd8#
   ```
   Could reward both correctness (short) and reasoning quality (long matches engine analysis).

### Evaluation

7. **Beyond Elo**: How to evaluate intermediate checkpoints without full games?
   - Puzzle accuracy by rating bucket?
   - Legal move rate?
   - Blunder rate (moves losing >100cp)?

8. **Engine calibration**: What Stockfish depth/settings for fair Elo estimation?

### Representation

9. **Vision models**: With Ministral's vision capability, is `FEN + board image` better than FEN alone?

10. **Move legality in output**: Should the model output move + confidence, or just move?

### Architecture

11. **Reasoning overhead**: Does CoT reasoning improve play quality enough to justify 10x token cost?

12. **Multi-task training**: Train single model for puzzles + games, or separate specialists?

## References

### Papers
- [DeepSeekMath: GRPO for mathematical reasoning](https://arxiv.org/abs/2402.03300) - Original GRPO paper
- [DeepSeek-R1](https://arxiv.org/abs/2501.12599) - GRPO for reasoning models
- [ChessLLM](https://arxiv.org/abs/2501.17186) - SFT approach reaching ~1788 Elo

### Libraries
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl/index) - GRPOTrainer, SFTTrainer
- [OpenEnv + GRPO notebook](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb) - Environment integration example
- [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero) - Minimal GRPO implementation

### Datasets
- [Lichess/chess-puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles) - 5.52M puzzles
- [pilipolio/lichess-puzzles-solutions](https://huggingface.co/datasets/pilipolio/lichess-puzzles-solutions) - Formatted puzzle dataset
- [Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games) - 7.14B games

### Articles
- [Demystifying Reasoning Models (GRPO deep-dive)](https://cameronrwolfe.substack.com/p/demystifying-reasoning-models)
- [Why GRPO is Important](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/)
- [PPO & GRPO comparison](https://yugeten.github.io/posts/2025/01/ppogrpo/)

### Examples

Mostly notebooks:
 - [SFT with LoRA/QLoRA using TRL](https://github.com/huggingface/trl/blob/794d87ff8d46fc41a07ffe9fb96771fee75922d4/examples/notebooks/sft_trl_lora_qlora.ipynb) on a range models (Qwen3, ...) 

 - [GRPO Qwen3-VL with QLoRA using TRL](https://github.com/huggingface/trl/blob/794d87ff8d46fc41a07ffe9fb96771fee75922d4/examples/notebooks/grpo_qwen3_vl.ipynb) on multi-modal math problems from [https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified]

 - [GRPO Ministral-3 with QLoRA using TRL]( https://github.com/huggingface/trl/blob/794d87ff8d46fc41a07ffe9fb96771fee75922d4/examples/notebooks/grpo_ministral3_vl.ipynb#L4)

---

## Experiment Log

### 2024-12-06: Qwen3-4B SFT on Mixed Tasks

**Setup:**
- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Dataset: `pilipolio/chess-mixed-tasks` (~9.5k train, ~1k test)
- Hardware: Modal A10G (24GB), ~3.3 hours
- Config: LoRA r=32, batch=8, grad_accum=2, lr=2e-4, cosine schedule, 3 epochs (726 steps with packing)

**Dataset composition:**
| Task | Count | Purpose |
|------|-------|---------|
| `puzzle_best_move` | 141 | Tactical pattern recognition |
| `puzzle_piece_positions` | 151 | Board awareness |
| `puzzle_legal_moves` | 131 | Rule understanding |
| `puzzle_legal_captures` | 100 | Tactical opportunities |
| `puzzle_ascii_board` | 127 | Visual representation |
| `puzzle_piece_captures` | 94 | Piece-specific captures |
| `toy_*` tasks | ~300 | Foundational FEN ↔ piece mapping |

**Results:**

| Metric | Start | End |
|--------|-------|-----|
| Eval loss | 2.05 | 0.17 |
| Token accuracy | 64% | 94% |
| Exact match (n=10) | 0% | 60% |

**Per-task accuracy (final):**
| Task | Accuracy | Notes |
|------|----------|-------|
| `puzzle_ascii_board` | 100% | Board visualization learned |
| `toy_fen_to_piece_list` | 100% | FEN parsing works |
| `puzzle_legal_moves` | 100% | Piece movement understood |
| `toy_fen_to_legal_moves_uci` | 50% | Partial |
| `puzzle_best_move` | 0% | Requires deeper tactics |
| `puzzle_piece_positions` | 0% | Subtle errors (missing pawns) |

**Error patterns observed:**
1. File confusion: `Qg4` vs `Qh4` (one square off)
2. Hallucinated captures: generates plausible but illegal moves
3. Missing pawns in piece lists

**Model:** [pilipolio/chess-puzzle-sft-qwen3-4b](https://huggingface.co/pilipolio/chess-puzzle-sft-qwen3-4b)

**Next steps:**
1. Host LoRA adapter for inference evaluation via `llm_evaluation.py`
2. Compare against baseline models (gpt-oss-20b, qwen3-32b, etc.)
3. Decide path forward:
   - **Option A:** More SFT on instruct model for board/tactical awareness
   - **Option B:** SFT + GRPO on reasoning model (e.g., Qwen3-4B with thinking tokens)

### 2024-12-06: Multi-Model Benchmark on Puzzle Tasks

**Setup:**
- Dataset: `pilipolio/lichess-puzzle-tasks` (test split, 58 examples)
- Evaluation: Parallel model evaluation via `asyncio.gather()` in `llm_evaluation.py`
- Metric: Exact match on best move prediction

**Models evaluated:**
| Model | Provider | Exact Match |
|-------|----------|-------------|
| openai/gpt-5-mini | OpenRouter | 56.9% |
| openai/gpt-oss-20b:free | OpenRouter | 41.4% |
| chess-puzzle (Qwen3-4B SFT) | Modal vLLM | 31.0% |
| qwen/qwen3-32b | OpenRouter | 17.2% |

**Observations:**
- GPT-5-mini leads, suggesting general reasoning capability helps with chess puzzles
- Fine-tuned chess-puzzle model (31.0%) underperforms larger general models
- Qwen3-32b surprisingly weak despite size - may need chess-specific tuning
- Parallelized benchmark runs 4x faster than sequential execution

**Changes made:**
- Added parallel model evaluation with `asyncio.gather()` to `run_benchmark()`
- Removed deprecated mistral model from benchmark config

---

## Next Phase: Reasoning Model Pipeline (Option B)

### Rationale

The instruct SFT achieved strong board awareness (100% on ASCII boards, legal moves) but 0% on best_move puzzles. This suggests the model pattern-matches rather than calculates. Reasoning models with native `<think>` tokens can externalize search, making tactical calculation more tractable.

**Concern:** Fine-tuning a reasoning model on chess might degrade general reasoning.
**Mitigation:** LoRA keeps base weights frozen; chess reasoning may transfer positively.

### Phase 2a: SFT on Reasoning Model

**Base model:** `Qwen/Qwen3-4B` (base, not instruct) or reasoning-tuned variant

**Dataset format with CoT:**

```python
def format_puzzle_with_reasoning(example):
    """Generate CoT from engine analysis or synthetic (GPT-5)."""
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Position: {example['fen']}\nFind the best move."
            },
            {
                "role": "assistant",
                "content": f"""<think>
{example['reasoning']}  # e.g., "The knight on c6 is pinned. Nxe5 wins material."
</think>
{example['best_move']}"""
            }
        ]
    }
```

**Reasoning data sources (in priority order):**
1. **Structured from puzzles:** Extract tactical themes, piece relationships from existing annotations
2. **Synthetic from GPT-5:** Generate explanations for puzzle solutions (validate with engine)
3. **Engine principal variation:** Convert Stockfish PV into natural language

#### Reasoning Data Generation Pipeline

Generate synthetic reasoning traces by providing an LLM (gpt-oss-20b or gpt-5-mini) with rich context from existing SFT tasks:

**Input context per puzzle:**
- FEN position
- Puzzle themes (e.g., `backRankMate`, `fork`, `pin`)
- Solution moves (full sequence, not just first move)
- Board state features from SFT tasks:
  - `puzzle_piece_positions`: All pieces with locations
  - `puzzle_legal_moves`: Legal moves for key pieces
  - `puzzle_legal_captures`: Available captures
  - `puzzle_ascii_board`: Visual board representation

**Prompt template:**
```
You are a chess instructor explaining puzzle solutions. Given:
- Position (FEN): {fen}
- Board:
{ascii_board}
- Pieces: {piece_positions}
- Themes: {themes}
- Solution: {solution_moves}

Write a concise reasoning trace (2-4 sentences) explaining WHY these moves work.
Focus on: checks, captures, threats, piece coordination, and the tactical pattern.
Do NOT just describe the moves - explain the forcing nature and why alternatives fail.

Output format:
<think>
[Your reasoning here]
</think>
{first_move}
```

**Example output:**
```
Position: 1r4k1/4nppp/8/4Pb2/8/1P5P/r1PR4/3R3K w - - 0 27
Themes: backRankMate
Solution: Rd8+ Rxd8 Rxd8#

<think>
Black's king is trapped on the back rank with no escape squares (g8 blocked by pawns).
Rd8+ forces Rxd8 (only legal response to check), then Rxd8# delivers mate.
The two rooks coordinate to exploit the weak back rank.
</think>
Rd8+
```

**Validation:** Parse generated reasoning, verify first move matches solution, check for hallucinated pieces/squares.

**Training config:**
```python
SFTConfig(
    per_device_train_batch_size=4,  # Longer sequences with reasoning
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=1024,  # Room for reasoning
    packing=False,    # CoT benefits from full attention
)
```

### Phase 2b: GRPO for Move Quality

**When to start GRPO:** After SFT achieves >50% legality rate on held-out positions.

**Reward function:**

```python
def chess_reasoning_reward(prompt: str, completion: str) -> float:
    fen = extract_fen(prompt)
    move = extract_final_move(completion)  # After </think> tag

    # Primary: move quality (legality + engine eval)
    move_score = chess_reward(fen, move)  # Returns [-1, 1]

    # Secondary: format compliance (weak signal)
    has_reasoning = "<think>" in completion and "</think>" in completion
    format_bonus = 0.05 if has_reasoning else 0.0

    # Optional: penalize empty or very short reasoning
    think_content = extract_think_content(completion)
    length_penalty = -0.1 if think_content and len(think_content) < 20 else 0.0

    return move_score + format_bonus + length_penalty
```

**Key insight:** Don't over-engineer reasoning rewards. Let GRPO discover which reasoning patterns lead to good moves. The primary signal is move quality.

**GRPO config:**
```python
GRPOConfig(
    learning_rate=1e-6,
    num_generations=8,       # 8 completions per position
    max_new_tokens=256,      # Cap reasoning length
    kl_coef=0.04,            # Prevent drift from SFT
    num_train_epochs=1,      # Single pass, monitor closely
)
```

### Evaluation Checkpoints

| Checkpoint | Metric | Target |
|------------|--------|--------|
| Pre-SFT baseline | Legality rate | Measure starting point |
| Post-SFT | Legality rate | >90% |
| Post-SFT | Best move accuracy (mate-in-1) | >50% |
| Post-SFT | Reasoning format compliance | >95% |
| Post-GRPO | Engine eval (avg centipawn loss) | <100 cp |
| Post-GRPO | Puzzle accuracy by rating | Track improvement curve |

### Open Questions

1. **Reasoning data quality:** How much does synthetic CoT quality matter? Compare GPT-5 vs structured templates.
2. **Thinking budget:** Should we constrain reasoning length or let GRPO discover optimal verbosity?
3. **Curriculum in GRPO:** Start with easier positions (lower-rated puzzles) or mixed difficulty?
