"""
Automated unit tests for GRPO loss and advantage normalizer.
Validates numerical stability, shape preservation, and autograd gradient flow.
"""

import pytest
import torch
from src.loss import (
    compute_group_advantages,
    compute_token_log_probs,
    grpo_surrogate_loss,
    compute_schulman_kl,
)


def test_group_advantage_normalization_zero_variance():
    """Verify that when all completions in a group have equal reward, advantage is strictly 0."""
    # 2 prompts, group size 4
    # Prompt 1: all 0 reward; Prompt 2: all 1 reward
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    advantages = compute_group_advantages(rewards, group_size=4)

    assert advantages.shape == rewards.shape
    assert torch.all(advantages == 0.0), f"Expected all zeros for zero variance, got {advantages}"


def test_group_advantage_normalization_values():
    """Verify standard z-score normalization within each independent group."""
    # 1 prompt, group size 4: rewards = [0, 0, 1, 1]
    # mean = 0.5, std = 0.5
    # advantages should be [-1, -1, 1, 1]
    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0])
    advantages = compute_group_advantages(rewards, group_size=4)

    expected = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    assert torch.allclose(advantages, expected, atol=1e-5)


def test_compute_token_log_probs_gradient_flow():
    """Verify that log-probs compute correctly and gradients backpropagate into logits."""
    N, L, V = 4, 8, 100
    logits = torch.randn(N, L, V, requires_grad=True)
    target_ids = torch.randint(0, V, (N, L))

    token_log_probs = compute_token_log_probs(logits, target_ids)

    assert token_log_probs.shape == (N, L)
    # Log probabilities must be strictly non-positive
    assert torch.all(token_log_probs <= 0.0)

    # Backpropagate a dummy scalar loss
    loss = token_log_probs.sum()
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert not torch.isinf(logits.grad).any()


def test_grpo_surrogate_loss_gradient_direction():
    """
    Fundamental RL Invariance:
    A positive advantage (A > 0) must encourage the policy to INCREASE the log prob of chosen tokens.
    Therefore, the gradient of loss w.r.t log_probs should be negative.
    """
    N, L = 2, 4
    log_probs = torch.zeros(N, L, requires_grad=True)
    old_log_probs = torch.zeros(N, L)
    advantages = torch.tensor([1.0, -1.0])  # Sequence 0 is rewarded, Sequence 1 is penalized
    mask = torch.ones(N, L)

    loss, metrics = grpo_surrogate_loss(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        completion_mask=mask,
        clip_eps=0.2,
    )

    loss.backward()

    # For sequence 0 (adv = +1.0), increasing log_prob lowers the loss -> gradient should be negative
    assert torch.all(log_probs.grad[0] < 0.0), "Gradient for positive advantage must be negative (descent increases prob)"
    # For sequence 1 (adv = -1.0), decreasing log_prob lowers the loss -> gradient should be positive
    assert torch.all(log_probs.grad[1] > 0.0), "Gradient for negative advantage must be positive (descent decreases prob)"


def test_schulman_kl_divergence_non_negative():
    """Schulman's k3 estimator must be identically 0 when policies match, and >= 0 otherwise."""
    N, L = 4, 10
    mask = torch.ones(N, L)

    # 1. Identical policies: KL should be 0.0
    log_probs = torch.randn(N, L)
    ref_log_probs = log_probs.clone()
    kl_identical = compute_schulman_kl(log_probs, ref_log_probs, mask)
    assert torch.isclose(kl_identical, torch.tensor(0.0), atol=1e-6)

    # 2. Perturbed policy: KL must be strictly positive
    perturbed_log_probs = log_probs + torch.randn(N, L) * 0.5
    kl_perturbed = compute_schulman_kl(perturbed_log_probs, ref_log_probs, mask)
    assert kl_perturbed.item() > 0.0, f"KL must be positive for divergent distributions, got {kl_perturbed.item()}"

