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

## The Reasoning Trace Pipeline

The key to effective SFT is generating high-quality synthetic reasoning traces that externalize the puzzle-solving process. The pipeline (`reasoning_generator.py`) produces structured traces with four design principles:

### 1. PGN Format as Bridge to RLVR

We use **PGN notation with interleaved comments** rather than natural language:

```
# Natural language (harder to verify)
"The rook takes on e7, threatening mate. Black must respond with Qb1 check..."

# PGN with comments (machine-verifiable)
25. Rxe7! {threatens mate on back rank} Qb1+ {forced - only check} 26. Nc1
```

This enables **Reinforcement Learning with Verifiable Rewards (RLVR)** because:
- Each move can be parsed and validated with python-chess
- Comments are clearly delimited with `{}`
- Refuted variations follow PGN spec: `(25. Qxb6? {loses to} Rxd1+ {back-rank})`

### 2. Theme Injection for Domain Grounding

Puzzle themes (fork, pin, backRankMate, skewer) are injected into prompts to:
- **Ground vocabulary** - Model learns domain-specific terminology
- **Prime pattern recognition** - Theme hints at the tactical motif to find
- **Enable balanced sampling** - `THEME_QUOTAS` prevent mate-puzzle overfit

### 3. The 5-Step Reasoning Structure

Each trace follows a structured format in `<think>` tags:

```
<think>
## Step 1: FEN parsing
Rank8: r...k → a8 rook, h8 king...

## Step 2: Piece Positions
White: Qh6, Re6, Nb3. Black: Kh8, Re7, Qb2

## Step 3: Position Summary
Material equal, Black king trapped on back rank after ...Rxd1

## Step 4: Candidate Moves
Qxh7+ reaches h7 along the b1-h7 diagonal. Rxe7 removes the defender...
Qxh7+, Rxe7, Rf8+

## Step 5: Lines Exploration
25. Rxe7! {wins the exchange} (25. Qxb6? {loses to} Rxd1+ Rxe1#)
</think>
25. Rxe7! Qb1+ 26. Nc1
```

### 4. Contrastive Learning via Refutations

The pipeline uses **Maia** (human-move predictor) + **Stockfish** to generate refutation lines:

```
Maia: "Humans would play Qxb6 (62%), Rxf6 (23%)..."
Stockfish: "Qxb6? loses to Rxd1+ Rxe1# (back-rank mate)"
```

This teaches the model not just "play Rd8+" but "don't play Qxb6 because of the back-rank threat".

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
3. **Format compliance** - 5-step reasoning structure present
4. **PGN validation** - verify ALL moves in solution, not just first move (enabled by structured PGN format)

The PGN format from the Reasoning Trace Pipeline enables multi-move verification: each move in the solution can be parsed and validated, providing richer signal than first-move-only rewards.

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

See [docs/experiment-log.md](experiment-log.md) for detailed experiment entries.

### Summary

| Date | Experiment | Key Result |
|------|------------|------------|
| 2024-12-06 | Qwen3-4B SFT on mixed tasks | 100% on board tasks, 0% on best_move |
| 2024-12-06 | Multi-model benchmark | GPT-5-mini 56.9%, fine-tuned 31.0% |
| 2024-12-09 | Reasoning SFT pipeline setup | 5-step format with PGN output |
| 2024-12-11 | gpt-4.1-nano vs gpt-5-nano | gpt-5-nano produces accurate reasoning |
| 2024-12-12 | GRPO debug with 4B SFT | Proper termination, ~20% first move accuracy |
| 2024-12-15 | 1.5k balanced dataset generation | 99.5% pass verification, Maia+Stockfish refutations |
| 2024-12-15 | SFT v2 on larger dataset | 0% first move, 80% legal, movement hallucinations |

**Current status:** SFT learns format (80% token acc) but not chess rules. Model hallucinates piece movements. Next: GRPO with legality rewards.
