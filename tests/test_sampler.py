"""
Automated unit tests for GRPO rollout sampler.
Validates prompt repeating, completion masking, and log-probability alignment.
"""

import pytest
import torch
import torch.nn as nn
from src.sampler import repeat_prompts_for_group, generate_group_rollouts


class MockCausalLM(nn.Module):
    """Minimal mock causal language model for fast, deterministic unit testing."""
    def __init__(self, vocab_size: int = 50, d_model: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # [B, L] -> [B, L, D] -> [B, L, V]
        x = self.embedding(input_ids)
        logits = self.linear(x)
        return logits


def test_repeat_prompts_for_group_dimensions():
    """Verify prompt repeating matches expected interleaved ordering and shape."""
    B, L, G = 2, 5, 4
    # Prompt 0 has all 1s, Prompt 1 has all 2s
    prompt_ids = torch.tensor([[1] * L, [2] * L])
    prompt_mask = torch.ones((B, L), dtype=torch.long)

    repeated_ids, repeated_mask = repeat_prompts_for_group(prompt_ids, prompt_mask, group_size=G)

    assert repeated_ids.shape == (B * G, L)
    assert repeated_mask.shape == (B * G, L)

    # First G rows must belong to Prompt 0
    assert torch.all(repeated_ids[:G] == 1)
    # Next G rows must belong to Prompt 1
    assert torch.all(repeated_ids[G:] == 2)


def test_generate_group_rollouts_shapes_and_masks():
    """Verify batched rollout generation produces correct completion masks."""
    vocab_size = 40
    model = MockCausalLM(vocab_size=vocab_size)
    
    B, L_prompt, G = 2, 4, 3
    max_new_tokens = 6
    prompt_ids = torch.randint(1, vocab_size - 1, (B, L_prompt))
    prompt_mask = torch.ones((B, L_prompt), dtype=torch.long)

    rollouts = generate_group_rollouts(
        model=model,
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        group_size=G,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        eos_token_id=None,
    )

    all_ids = rollouts["all_ids"]
    mask = rollouts["completion_mask"]
    old_log_probs = rollouts["old_log_probs"]

    N = B * G
    L_total = L_prompt + max_new_tokens

    # Verify tensor shapes
    assert all_ids.shape == (N, L_total)
    assert mask.shape == (N, L_total)
    assert old_log_probs.shape == (N, L_total)

    # Prompt portion of completion_mask must be strictly 0.0
    assert torch.all(mask[:, :L_prompt] == 0.0), "Prompt tokens must not be included in completion mask"
    # Generated portion of completion_mask must be 1.0 (no early EOS specified)
    assert torch.all(mask[:, L_prompt:] == 1.0), "Generated tokens must have mask = 1.0"

    # Log probabilities on active tokens must be <= 0.0
    active_log_probs = old_log_probs[:, L_prompt:]
    assert torch.all(active_log_probs <= 0.0)
    assert not torch.isnan(active_log_probs).any()
    assert not torch.isinf(active_log_probs).any()


def test_generate_group_rollouts_eos_masking():
    """Verify that hitting EOS token properly masks subsequent padding tokens to 0.0."""
    vocab_size = 20
    model = MockCausalLM(vocab_size=vocab_size)
    
    # Force the model to always predict EOS token (token ID 2)
    eos_id = 2
    with torch.no_grad():
        model.linear.bias.fill_(-100.0)
        model.linear.bias[eos_id] = 100.0

    B, L_prompt, G = 1, 3, 2
    prompt_ids = torch.randint(3, vocab_size, (B, L_prompt))
    prompt_mask = torch.ones((B, L_prompt), dtype=torch.long)

    rollouts = generate_group_rollouts(
        model=model,
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        group_size=G,
        max_new_tokens=5,
        eos_token_id=eos_id,
        pad_token_id=0,
    )

    mask = rollouts["completion_mask"]
    all_ids = rollouts["all_ids"]

    # Since the first generated token was EOS, the loop should early stop or subsequent tokens masked to 0
    # First token after prompt should be EOS
    assert torch.all(all_ids[:, L_prompt] == eos_id)
    # The mask on the EOS token itself is 1.0 (it's the completion end), any padded tokens after must be 0.0
    if mask.shape[1] > L_prompt + 1:
        assert torch.all(mask[:, L_prompt + 1:] == 0.0)

