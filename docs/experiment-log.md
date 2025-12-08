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
