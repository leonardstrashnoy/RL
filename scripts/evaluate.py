#!/usr/bin/env python3
"""
Evaluation Script for 2048 Game Strategy

Benchmarks a trained model against random and heuristic baselines.

Usage:
    python scripts/evaluate.py --model saved_model/lora_adapter --games 100
    python scripts/evaluate.py --baseline-only --games 50
"""

import argparse
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict
from collections import defaultdict

import torch
from peft import PeftModel
from unsloth import FastLanguageModel, check_python_modules, create_locked_down_function, execute_with_time_limit


# ============================================================================
# 2048 Game Implementation (same as train.py)
# ============================================================================

def _compress_and_merge_row_left(row: List[int]) -> Tuple[List[int], int, bool]:
    n = len(row)
    tiles = [x for x in row if x != 0]
    gained = 0
    i = 0
    merged = []
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            v = tiles[i] * 2
            gained += v
            merged.append(v)
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged += [0] * (n - len(merged))
    changed = merged != row
    return merged, gained, changed


def _move_left(board):
    new_board, total_gain, changed_any = [], 0, False
    for row in board:
        new_row, gained, changed = _compress_and_merge_row_left(row)
        new_board.append(new_row)
        total_gain += gained
        changed_any = changed_any or changed
    return new_board, total_gain, changed_any


def _move_right(board):
    new_board, total_gain, changed_any = [], 0, False
    for row in board:
        rev = list(reversed(row))
        new_rev, gained, changed = _compress_and_merge_row_left(rev)
        new_board.append(list(reversed(new_rev)))
        total_gain += gained
        changed_any = changed_any or changed
    return new_board, total_gain, changed_any


def _transpose(board):
    return [list(row) for row in zip(*board)]


def _move_up(board):
    t = _transpose(board)
    moved, gain, changed = _move_left(t)
    return _transpose(moved), gain, changed


def _move_down(board):
    t = _transpose(board)
    moved, gain, changed = _move_right(t)
    return _transpose(moved), gain, changed


def _empty_cells(board):
    size = len(board)
    return [(r, c) for r in range(size) for c in range(size) if board[r][c] == 0]


def _can_move(board):
    if _empty_cells(board):
        return True
    size = len(board)
    for r in range(size):
        for c in range(size - 1):
            if board[r][c] == board[r][c + 1]:
                return True
    for r in range(size - 1):
        for c in range(size):
            if board[r][c] == board[r + 1][c]:
                return True
    return False


@dataclass
class GameBoard:
    size: int
    seed: Optional[int] = None
    target: int = 2048
    probability_fours: float = 0.10
    _rng: random.Random = field(init=False, repr=False)
    _board: List[List[int]] = field(init=False, repr=False)
    _score: int = field(default=0, init=False, repr=False)
    _state: str = field(default="ongoing", init=False, repr=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self._add_random_tile()
        self._add_random_tile()
        self._update_state_after_change()

    def board(self): return [row[:] for row in self._board]
    def state(self): return self._state
    def score(self): return self._score
    def max_tile(self): return max(max(row) for row in self._board)

    def do_action(self, key: str):
        if self._state != "ongoing":
            return
        if not isinstance(key, str) or len(key) == 0:
            self._state = "failed"
            return
        k = key.strip().lower()
        move_map = {"a": _move_left, "d": _move_right, "w": _move_up, "s": _move_down}
        if k not in move_map:
            self._state = "failed"
            return
        new_board, gain, changed = move_map[k](self._board)
        if changed:
            self._board = new_board
            self._score += gain
            self._add_random_tile()
        self._update_state_after_change()

    def _add_random_tile(self):
        empties = _empty_cells(self._board)
        if not empties:
            return False
        r, c = self._rng.choice(empties)
        self._board[r][c] = 4 if self._rng.random() < self.probability_fours else 2
        return True

    def _update_state_after_change(self):
        if any(self.target in row for row in self._board):
            self._state = "success"
        elif not _can_move(self._board):
            self._state = "failed"


# ============================================================================
# Baseline Strategies
# ============================================================================

def random_strategy(board):
    """Random baseline - picks random valid move."""
    return random.choice(["W", "A", "S", "D"])


def corner_strategy(board):
    """Heuristic - tries to keep largest tile in corner."""
    # Priority: Down, Right, Left, Up (keeps tiles in bottom-right)
    for move in ["S", "D", "A", "W"]:
        return move
    return "S"


def cycle_strategy(board):
    """Cycles through moves in a pattern."""
    # Classic pattern that works decently
    moves = ["S", "D", "S", "A"]
    if not hasattr(cycle_strategy, "idx"):
        cycle_strategy.idx = 0
    move = moves[cycle_strategy.idx % len(moves)]
    cycle_strategy.idx += 1
    return move


# ============================================================================
# Model Strategy
# ============================================================================

PROMPT = """
Create a new short 2048 strategy using only native Python code.
You are given a list of list of numbers for the current board state.
Output one action for "W", "A", "S", "D" on what is the optimal next step.
Output your new short function in backticks using the format below:
```python
def strategy(board):
    return "W" # Example
```
All helper functions should be inside def strategy. Only output the short function `strategy`.
""".strip()


def extract_function(text: str) -> Optional[str]:
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def strategy(board):"):
            return fx
    return None


def load_model(model_path: str, base_model: str = "unsloth/gpt-oss-20b"):
    """Load trained model with LoRA adapter."""
    print(f"Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=768,
        load_in_4bit=True,
        offload_embedding=True,
    )

    print(f"Loading LoRA adapter: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def generate_strategy(model, tokenizer) -> Optional[Callable]:
    """Generate a strategy function from the model."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=1.0,
            max_new_tokens=512,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    function_text = extract_function(response)

    if function_text is None:
        return None

    ok, info = check_python_modules(function_text)
    if not ok or "error" in info:
        return None

    try:
        return create_locked_down_function(function_text)
    except:
        return None


# ============================================================================
# Evaluation
# ============================================================================

@execute_with_time_limit(10)
def run_game(strategy: Callable, seed: int, board_size: int = 6) -> Dict:
    """Run a single game and return results."""
    game = GameBoard(size=board_size, seed=seed, target=2048, probability_fours=0.10)
    steps = 0
    max_steps = 10000

    while game.state() == "ongoing" and steps < max_steps:
        try:
            action = strategy(game.board())
            if not isinstance(action, str):
                break
            game.do_action(action)
            steps += 1
        except:
            break

    return {
        "state": game.state(),
        "score": game.score(),
        "max_tile": game.max_tile(),
        "steps": steps,
        "success": game.state() == "success",
    }


def evaluate_strategy(
    strategy: Callable,
    name: str,
    num_games: int = 100,
    board_size: int = 6,
    verbose: bool = True,
) -> Dict:
    """Evaluate a strategy over multiple games."""
    results = []
    successes = 0
    timeouts = 0

    for i in range(num_games):
        seed = i * 1000 + 42  # Reproducible seeds

        try:
            result = run_game(strategy, seed, board_size)
            results.append(result)
            if result["success"]:
                successes += 1
        except TimeoutError:
            timeouts += 1
            results.append({
                "state": "timeout",
                "score": 0,
                "max_tile": 0,
                "steps": 0,
                "success": False,
            })

        if verbose and (i + 1) % 10 == 0:
            print(f"  {name}: {i + 1}/{num_games} games, {successes} wins")

    scores = [r["score"] for r in results]
    max_tiles = [r["max_tile"] for r in results]
    steps = [r["steps"] for r in results if r["steps"] > 0]

    return {
        "name": name,
        "games": num_games,
        "wins": successes,
        "win_rate": successes / num_games * 100,
        "timeouts": timeouts,
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "avg_max_tile": sum(max_tiles) / len(max_tiles) if max_tiles else 0,
        "highest_tile": max(max_tiles) if max_tiles else 0,
        "avg_steps": sum(steps) / len(steps) if steps else 0,
    }


def print_results(results: List[Dict]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    headers = ["Strategy", "Games", "Wins", "Win%", "Timeouts", "Avg Score", "Max Tile", "Avg Steps"]
    row_format = "{:<15} {:>6} {:>6} {:>6.1f} {:>8} {:>10.0f} {:>9} {:>10.0f}"
    header_format = "{:<15} {:>6} {:>6} {:>6} {:>8} {:>10} {:>9} {:>10}"

    print(header_format.format(*headers))
    print("-" * 80)

    for r in results:
        print(row_format.format(
            r["name"],
            r["games"],
            r["wins"],
            r["win_rate"],
            r["timeouts"],
            r["avg_score"],
            r["highest_tile"],
            r["avg_steps"],
        ))

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate 2048 strategies")
    parser.add_argument("--model", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--base_model", default="unsloth/gpt-oss-20b", help="Base model")
    parser.add_argument("--games", type=int, default=100, help="Number of games per strategy")
    parser.add_argument("--board_size", type=int, default=6, help="Board size")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baselines")
    parser.add_argument("--model-samples", type=int, default=5, help="Number of strategies to sample from model")

    args = parser.parse_args()

    results = []

    # Baseline evaluations
    print("\nEvaluating baselines...")

    print("\n[1/3] Random Strategy")
    results.append(evaluate_strategy(random_strategy, "Random", args.games, args.board_size))

    print("\n[2/3] Corner Strategy")
    results.append(evaluate_strategy(corner_strategy, "Corner", args.games, args.board_size))

    print("\n[3/3] Cycle Strategy")
    cycle_strategy.idx = 0  # Reset
    results.append(evaluate_strategy(cycle_strategy, "Cycle", args.games, args.board_size))

    # Model evaluation
    if args.model and not args.baseline_only:
        print(f"\nLoading trained model from {args.model}...")
        model, tokenizer = load_model(args.model, args.base_model)

        print(f"\nGenerating {args.model_samples} strategies from model...")
        model_wins = 0
        model_games = 0

        for i in range(args.model_samples):
            print(f"\n[Model Sample {i + 1}/{args.model_samples}]")
            strategy = generate_strategy(model, tokenizer)

            if strategy is None:
                print("  Failed to generate valid strategy")
                continue

            result = evaluate_strategy(
                strategy,
                f"Model-{i + 1}",
                args.games // args.model_samples,
                args.board_size,
            )
            results.append(result)
            model_wins += result["wins"]
            model_games += result["games"]

        # Aggregate model results
        if model_games > 0:
            results.append({
                "name": "Model (avg)",
                "games": model_games,
                "wins": model_wins,
                "win_rate": model_wins / model_games * 100,
                "timeouts": sum(r["timeouts"] for r in results if r["name"].startswith("Model-")),
                "avg_score": sum(r["avg_score"] * r["games"] for r in results if r["name"].startswith("Model-")) / model_games,
                "max_score": max(r["max_score"] for r in results if r["name"].startswith("Model-")),
                "avg_max_tile": sum(r["avg_max_tile"] * r["games"] for r in results if r["name"].startswith("Model-")) / model_games,
                "highest_tile": max(r["highest_tile"] for r in results if r["name"].startswith("Model-")),
                "avg_steps": sum(r["avg_steps"] * r["games"] for r in results if r["name"].startswith("Model-")) / model_games,
            })

    print_results(results)


if __name__ == "__main__":
    main()
