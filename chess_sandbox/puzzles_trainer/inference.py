"""Inference utilities for chess validation with thinking budget control.

References:
- Zach Mueller's TIL: https://muellerzr.github.io/til/end_thinking.html
- vLLM thinking budget PR: https://github.com/vllm-project/vllm/pull/20859
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch
from transformers.generation.logits_process import LogitsProcessor

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class ThinkingTokenBudgetProcessor(LogitsProcessor):
    """Limit thinking tokens before forcing </think>.

    At 95% of budget: boosts </think> probability
    At budget limit: forces </think> token
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_thinking_tokens: int = 100):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.think_end_token = tokenizer.encode("</think>", add_special_tokens=False)[0]  # pyright: ignore[reportUnknownMemberType]
        self.nl_token = tokenizer.encode("\n", add_special_tokens=False)[0]  # pyright: ignore[reportUnknownMemberType]
        self.tokens_generated = 0
        self.stopped_thinking = False
        self.neg_inf = float("-inf")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.tokens_generated += 1

        if self.max_thinking_tokens == 0 and not self.stopped_thinking:
            scores[:] = self.neg_inf
            scores[0][self.nl_token] = 0
            scores[0][self.think_end_token] = 0
            self.stopped_thinking = True
            return scores

        if not self.stopped_thinking:
            ratio = self.tokens_generated / self.max_thinking_tokens

            # Boost </think> probability near budget
            if ratio > 0.95:
                boost = 1 + ratio
                scores[0][self.nl_token] *= boost
                scores[0][self.think_end_token] *= boost

            # Force end at budget limit
            if self.tokens_generated >= self.max_thinking_tokens - 1:
                scores[:] = self.neg_inf
                if self.tokens_generated == self.max_thinking_tokens - 1:
                    scores[0][self.nl_token] = 0
                else:
                    scores[0][self.think_end_token] = 0
                    self.stopped_thinking = True

        return scores


def batch_generate(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int | None = None,
    max_thinking_tokens: int | None = None,
    batch_size: int = 8,
) -> list[str]:
    """Generate outputs with optional thinking budget.

    Args:
        model: HuggingFace model for generation
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of prompts to generate from
        max_new_tokens: Maximum tokens to generate (default 256)
        max_thinking_tokens: Max tokens for thinking (None to disable limit)
        batch_size: Batch size for generation

    Returns:
        List of generated outputs with think tags stripped
    """
    if max_new_tokens is None:
        max_new_tokens = 256

    all_outputs: list[str] = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]

            # Use left-padding for decoder-only models during generation
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)  # pyright: ignore[reportUnknownMemberType]
            tokenizer.padding_side = original_padding_side
            inputs = {k: v.to(model.device) for k, v in inputs.items()}  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

            # Build logits processors
            logits_processor: list[LogitsProcessor] = []
            if max_thinking_tokens is not None:
                logits_processor.append(ThinkingTokenBudgetProcessor(tokenizer, max_thinking_tokens))

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,  # pyright: ignore[reportUnknownMemberType]
                logits_processor=logits_processor if logits_processor else None,
            )

            # Decode only the generated tokens
            for j, output in enumerate(outputs):
                input_len = int(inputs["input_ids"][j].shape[0])  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)  # pyright: ignore[reportUnknownMemberType]
                generated = strip_think_tags(generated)
                all_outputs.append(generated)

    return all_outputs


def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> blocks from model output.

    Handles truncated outputs gracefully:
    - Complete blocks: removed entirely
    - Unclosed <think> at end: stripped from <think> onwards
    - Orphaned </think> at start: stripped through </think>

    >>> strip_think_tags("hello")
    'hello'
    >>> strip_think_tags("<think>reasoning</think>answer")
    'answer'
    >>> strip_think_tags("<think>reasoning</think>\\n\\nanswer")
    'answer'
    >>> strip_think_tags("prefix<think>middle</think>suffix")
    'prefixsuffix'
    >>> strip_think_tags("<think>block1</think>mid<think>block2</think>end")
    'midend'
    >>> strip_think_tags("<think>unclosed")
    ''
    >>> strip_think_tags("orphan</think>text")
    'text'
    >>> strip_think_tags("answer<think>still thinking")
    'answer'
    """
    # First, remove any complete <think>...</think> blocks
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Handle truncated output: unclosed <think> at end
    result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL)

    # Handle edge case: orphaned </think> at start
    result = re.sub(r"^.*?</think>", "", result, flags=re.DOTALL)

    return result.strip()
