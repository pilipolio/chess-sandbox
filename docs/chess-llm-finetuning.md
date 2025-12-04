# Fine-tuning Small LLMs for Chess

A two-stage training pipeline combining Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) to teach small language models to play chess.

## Approach Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Stage 1: SFT                        Stage 2: GRPO                         │
│   ───────────────                     ─────────────────                     │
│   Structured tasks with               Open-ended positions with             │
│   single correct answers              multiple acceptable moves             │
│                                                                             │
│   • Legal move detection              • Play against verification           │
│   • Mate-in-1 puzzles                 • Engine-scored rewards               │
│   • Forced sequences                  • Legality constraints                │
│                                                                             │
│   Dataset: Lichess puzzles            Reward: Stockfish eval + legality     │
│   Tool: TRL SFTTrainer                Tool: TRL GRPOTrainer                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Motivation:** Even strong reasoning models like GPT-4o and o1 lack chess board awareness and frequently produce illegal moves. LLMs, like humans, benefit from drilling patterns and board visualization ("System 1" intuition) before developing calculation and strategic understanding ("System 2" reasoning). The two-stage approach grounds the model in legal chess patterns through SFT, then refines play quality and look-ahead capabilities using verifiable rewards (GRPO).

## Stage 1: Supervised Fine-Tuning (SFT)

### Tasks

| Task | Input | Output | Purpose |
|------|-------|--------|---------|
| Board ↔ FEN | ASCII board diagram | FEN string (or vice versa) | Board visualization and spatial reasoning |
| Legal moves | FEN position | List of legal moves | Learn chess rules implicitly |
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

## Stage 2: GRPO Reinforcement Learning

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
| **[Ministral-3B](https://huggingface.co/mistralai/Ministral-3b-instruct)** | 3B | Vision capability for board images |

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

### Evaluation

5. **Beyond Elo**: How to evaluate intermediate checkpoints without full games?
   - Puzzle accuracy by rating bucket?
   - Legal move rate?
   - Blunder rate (moves losing >100cp)?

6. **Engine calibration**: What Stockfish depth/settings for fair Elo estimation?

### Representation

7. **Vision models**: With Ministral's vision capability, is `FEN + board image` better than FEN alone?

8. **Move legality in output**: Should the model output move + confidence, or just move?

### Architecture

9. **Reasoning overhead**: Does CoT reasoning improve play quality enough to justify 10x token cost?

10. **Multi-task training**: Train single model for puzzles + games, or separate specialists?

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
