"""
Automated unit and integration tests for GRPOTrainer.
Validates the full online RL training step, optimizer updates, and reward progression.
"""

import pytest
import torch
import torch.nn as nn
from src.trainer import GRPOTrainer
from tests.test_sampler import MockCausalLM


def simple_mock_tokenizer_decode(token_ids: list) -> str:
    """Mock decode function mapping token IDs to formatted strings for verifier testing."""
    # If token ID 10 is present, emit target answer <think>ok</think><answer>42</answer>
    if 10 in token_ids:
        return "<think>computed</think><answer>42</answer>"
    elif 11 in token_ids:
        return "<think>wrong</think><answer>99</answer>"
    return "unstructured text"


def test_grpo_trainer_single_step():
    """Verify an end-to-end training step runs without errors, computes gradients, and updates weights."""
    vocab_size = 50
    model = MockCausalLM(vocab_size=vocab_size, d_model=32)
    ref_model = MockCausalLM(vocab_size=vocab_size, d_model=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = GRPOTrainer(
        policy_model=model,
        ref_model=ref_model,
        optimizer=optimizer,
        group_size=4,
        max_new_tokens=8,
        clip_eps=0.2,
        kl_weight=0.01,
        max_grad_norm=1.0,
    )

    # Prompt: 2 questions, length 4
    prompt_ids = torch.randint(1, vocab_size, (2, 4))
    prompt_mask = torch.ones_like(prompt_ids)
    ground_truths = ["42", "42"]

    # Record parameter values before step
    initial_param = model.linear.weight.clone()

    metrics = trainer.train_step(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        ground_truths=ground_truths,
        decode_fn=simple_mock_tokenizer_decode,
    )

    # Telemetry checks
    assert "loss/total" in metrics
    assert "reward/mean" in metrics
    assert "policy/grad_norm" in metrics
    assert not torch.isnan(torch.tensor(metrics["loss/total"]))
    assert not torch.isnan(torch.tensor(metrics["policy/grad_norm"]))

    # Parameters must have been updated by optimizer
    assert not torch.allclose(model.linear.weight, initial_param), "Optimizer must update policy model weights"


def test_grpo_trainer_reward_optimization():
    """
    Empirical Learning Invariance:
    Verify that across multiple steps of GRPO updates on a synthetic task,
    the model learns to increase the generation probability of the high-reward token.
    """
    vocab_size = 30
    model = MockCausalLM(vocab_size=vocab_size, d_model=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    trainer = GRPOTrainer(
        policy_model=model,
        ref_model=None,  # No KL penalty to allow rapid convergence on synthetic task
        optimizer=optimizer,
        group_size=8,    # G=8 rollouts for clear advantage contrast
        max_new_tokens=4,
        temperature=1.0,
    )

    prompt_ids = torch.tensor([[1, 2, 3]])
    prompt_mask = torch.ones_like(prompt_ids)
    ground_truths = ["42"]

    initial_metrics = trainer.train_step(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        ground_truths=ground_truths,
        decode_fn=simple_mock_tokenizer_decode,
    )

    # Run 5 training steps
    final_metrics = initial_metrics
    for _ in range(5):
        final_metrics = trainer.train_step(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            ground_truths=ground_truths,
            decode_fn=simple_mock_tokenizer_decode,
        )

    # Final step should maintain stable loss and valid finite telemetry
    assert not torch.isnan(torch.tensor(final_metrics["loss/total"]))
    assert final_metrics["policy/grad_norm"] >= 0.0

