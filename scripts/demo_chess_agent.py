#!/usr/bin/env python3
"""
Demo script for the pydantic_ai chess agent.

Example usage:
    python scripts/demo_chess_agent.py
    python scripts/demo_chess_agent.py --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    python scripts/demo_chess_agent.py --query "What are the best moves in this position?"
"""

import asyncio
from argparse import ArgumentParser

from chess_sandbox.chess_agent import create_chess_agent


async def run_agent_query(query: str, fen: str | None = None) -> None:
    """Run a single query against the chess agent."""
    agent, deps = create_chess_agent(fen=fen)

    print(f"\n🏁 Starting position: {deps.board.fen()}")
    print(f"📝 Query: {query}\n")

    result = await agent.run(query, deps=deps)
    print(f"♟️  Agent response:\n{result.data}\n")

    deps.engine.quit()


async def interactive_mode(fen: str | None = None) -> None:
    """Run the agent in interactive mode."""
    agent, deps = create_chess_agent(fen=fen)

    print("♟️  Pydantic AI Chess Agent - Interactive Mode")
    print(f"🏁 Starting position: {deps.board.fen()}")
    print("\nType your questions or commands. Type 'quit' or 'exit' to exit.\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            if not user_input:
                continue

            result = await agent.run(user_input, deps=deps)
            print(f"\nAgent: {result.data}\n")

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        deps.engine.quit()


def main() -> None:
    parser = ArgumentParser(description="Demo Pydantic AI Chess Agent")
    parser.add_argument(
        "--fen",
        type=str,
        help="Starting position in FEN notation (default: standard starting position)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to run (if not provided, enters interactive mode)",
    )

    args = parser.parse_args()

    if args.query:
        asyncio.run(run_agent_query(args.query, args.fen))
    else:
        asyncio.run(interactive_mode(args.fen))


if __name__ == "__main__":
    main()
