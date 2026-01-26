#!/usr/bin/env python3
"""
GRPO Training Script for 2048 Game Strategy

Trains GPT-OSS-20B with LoRA to generate winning 2048 strategies
using Group Relative Policy Optimization (GRPO).

Usage:
    python scripts/train.py --max_steps 1000 --save_steps 100
    python scripts/train.py --resume outputs/checkpoint-100
"""

import argparse
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable

import numpy as np
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, check_python_modules, create_locked_down_function, execute_with_time_limit


# ============================================================================
# 2048 Game Implementation
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


def _move_left(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
    changed_any = False
    total_gain = 0
    new_board = []
    for row in board:
        new_row, gained, changed = _compress_and_merge_row_left(row)
        new_board.append(new_row)
        total_gain += gained
        changed_any = changed_any or changed
    return new_board, total_gain, changed_any


def _move_right(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
    changed_any = False
    total_gain = 0
    new_board = []
    for row in board:
        rev = list(reversed(row))
        new_rev, gained, changed = _compress_and_merge_row_left(rev)
        new_row = list(reversed(new_rev))
        new_board.append(new_row)
        total_gain += gained
        changed_any = changed_any or changed
    return new_board, total_gain, changed_any


def _transpose(board: List[List[int]]) -> List[List[int]]:
    return [list(row) for row in zip(*board)]


def _move_up(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
    t = _transpose(board)
    moved, gain, changed = _move_left(t)
    return _transpose(moved), gain, changed


def _move_down(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
    t = _transpose(board)
    moved, gain, changed = _move_right(t)
    return _transpose(moved), gain, changed


def _empty_cells(board: List[List[int]]) -> List[Tuple[int, int]]:
    size = len(board)
    return [(r, c) for r in range(size) for c in range(size) if board[r][c] == 0]


def _can_move(board: List[List[int]]) -> bool:
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
        if self.size < 2:
            raise ValueError("Board size must be at least 2.")
        self._rng = random.Random(self.seed)
        self._board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self._add_random_tile()
        self._add_random_tile()
        self._update_state_after_change()

    def board(self) -> List[List[int]]:
        return [row[:] for row in self._board]

    def state(self) -> str:
        return self._state

    def score(self) -> int:
        return self._score

    def do_action(self, key: str) -> None:
        if self._state != "ongoing":
            return
        if not isinstance(key, str) or len(key) == 0:
            self._state = "failed"
            return
        k = key.strip().lower()
        if k == "q":
            self._state = "failed"
            return
        move_map = {"a": _move_left, "d": _move_right, "w": _move_up, "s": _move_down}
        if k not in move_map:
            self._state = "failed"
            return
        mover = move_map[k]
        new_board, gain, changed = mover(self._board)
        if changed:
            self._board = new_board
            self._score += gain
            self._add_random_tile()
        self._update_state_after_change()

    def _add_random_tile(self) -> bool:
        empties = _empty_cells(self._board)
        if not empties:
            return False
        r, c = self._rng.choice(empties)
        self._board[r][c] = 4 if self._rng.random() < self.probability_fours else 2
        return True

    def _update_state_after_change(self) -> None:
        if any(self.target in row for row in self._board):
            self._state = "success"
            return
        if not _can_move(self._board):
            self._state = "failed"
            return
        self._state = "ongoing"


# ============================================================================
# Strategy Execution
# ============================================================================

def _execute_strategy(strategy: Callable, game: GameBoard):
    assert callable(strategy)
    steps = 0
    while game.state() == "ongoing":
        action = strategy(game.board())
        steps += 1
        if type(action) is not str:
            return steps, "failed"
        game.do_action(action)
    return steps, game.state()


@execute_with_time_limit(5)
def execute_strategy(strategy: Callable, game: GameBoard):
    return _execute_strategy(strategy, game)


# ============================================================================
# Reward Functions
# ============================================================================

def extract_function(text: str) -> Optional[str]:
    """Extract Python function from markdown code block."""
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def strategy(board):"):
            return fx
    return None


def function_works(completions, **kwargs):
    """Reward for valid Python syntax."""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_python_modules(function)
        if function is None or "error" in info:
            score = -2.0
        else:
            try:
                create_locked_down_function(function)
                score = 1.0
            except:
                score = -0.5
        scores.append(score)
    return scores


def no_cheating(completions, **kwargs):
    """Penalize importing non-stdlib modules."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_python_modules(function)
            scores.append(1.0 if ok else -20.0)
        else:
            scores.append(-1.0)
    return scores


# Global counter for printing
PRINTER = 0


def strategy_succeeds(completions, **kwargs):
    """Reward for successfully reaching 2048."""
    global PRINTER
    scores = []
    seed = np.random.randint(10000)

    for completion in completions:
        printed = False
        response = completion[0]["content"]
        function = extract_function(response)

        if PRINTER % 5 == 0:
            printed = True
            print(function)
        PRINTER += 1

        if function is not None:
            ok, info = check_python_modules(function)
        if function is None or "error" in info:
            scores.append(0)
            continue

        try:
            new_strategy = create_locked_down_function(function)
        except:
            scores.append(0)
            continue

        try:
            game = GameBoard(size=6, seed=seed, target=2048, probability_fours=0.10)
            steps, game_state = execute_strategy(new_strategy, game)
            print(f"Steps = {steps} State = {game_state}")
            if not printed:
                print(function)
            if game_state == "success":
                scores.append(20.0)
            else:
                scores.append(2.0)
        except TimeoutError:
            print("Timeout")
            scores.append(-1.0)
        except Exception as e:
            print(f"Exception = {str(e)}")
            scores.append(-3.0)

    return scores


# ============================================================================
# Training
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


def create_dataset(num_examples: int = 1000) -> Dataset:
    """Create training dataset with prompt replicas."""
    return Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": PROMPT}],
            "answer": 0,
            "reasoning_effort": "low"
        }
    ] * num_examples)


def load_model(
    model_name: str = "unsloth/gpt-oss-20b",
    max_seq_length: int = 768,
    lora_rank: int = 4,
    load_in_4bit: bool = True,
):
    """Load model with LoRA configuration."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        offload_embedding=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    return model, tokenizer


def train(
    model,
    tokenizer,
    dataset: Dataset,
    max_steps: int = 1000,
    save_steps: int = 100,
    output_dir: str = "outputs",
    resume_from: Optional[str] = None,
    learning_rate: float = 5e-5,
    batch_size: int = 1,
    num_generations: int = 2,
):
    """Run GRPO training."""
    max_prompt_length = len(tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True
    )) + 1
    max_completion_length = 768 - max_prompt_length

    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        save_steps=save_steps,
        report_to="none",
        output_dir=output_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            function_works,
            no_cheating,
            strategy_succeeds,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train(resume_from_checkpoint=resume_from)

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train GPT-OSS on 2048 with GRPO")
    parser.add_argument("--model", default="unsloth/gpt-oss-20b", help="Base model name")
    parser.add_argument("--max_seq_length", type=int, default=768, help="Max sequence length")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--max_steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--num_generations", type=int, default=2, help="Generations per prompt")
    parser.add_argument("--num_examples", type=int, default=1000, help="Dataset size")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Use 4-bit quantization")

    args = parser.parse_args()

    print("=" * 60)
    print("GRPO Training for 2048 Game Strategy")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output: {args.output_dir}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        load_in_4bit=args.load_in_4bit,
    )

    # Create dataset
    print("\nCreating dataset...")
    dataset = create_dataset(args.num_examples)

    # Train
    print("\nStarting training...")
    trainer = train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        resume_from=args.resume,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
    )

    # Save final model
    print("\nSaving final model...")
    model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
