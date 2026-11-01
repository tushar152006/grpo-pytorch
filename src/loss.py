"""
Group Relative Policy Optimization (GRPO) Core Loss & Advantage Module.
Strict native PyTorch implementation without high-level library dependencies.
Reference: DeepSeek-R1 (2025), Schulman (2020) Unbiased KL Divergence.
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes group-relative normalized advantages.
    
    Formula:
        A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
    
    Args:
        rewards: 1D Tensor of shape [N] where N = B * G (total completions),
                 or 2D Tensor of shape [B, G].
        group_size: Number of rollouts sampled per prompt (G).
        eps: Epsilon guard for numerical stability.
        
    Returns:
        advantages: Tensor of shape [N] matching input rewards shape.
    """
    original_shape = rewards.shape
    if rewards.dim() == 1:
        num_prompts = rewards.shape[0] // group_size
        rewards_2d = rewards.view(num_prompts, group_size)
    elif rewards.dim() == 2:
        rewards_2d = rewards
    else:
        raise ValueError(f"Expected 1D or 2D rewards tensor, got shape {original_shape}")

    # Compute mean and unbiased standard deviation within each group: shape [B, 1]
    group_mean = rewards_2d.mean(dim=-1, keepdim=True)
    group_std = rewards_2d.std(dim=-1, keepdim=True, unbiased=False)

    # Edge case: If all completions in a group have identical rewards (std ~ 0),
    # advantage should be strictly 0 to prevent amplifying numerical jitter.
    zero_var_mask = group_std < 1e-7
    advantages_2d = (rewards_2d - group_mean) / (group_std + eps)
    advantages_2d = torch.where(zero_var_mask, torch.zeros_like(advantages_2d), advantages_2d)

    return advantages_2d.view(original_shape)


def compute_token_log_probs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Gathers per-token log probabilities using log_softmax and vectorized index gathering.
    
    Args:
        logits: Tensor of shape [N, L, V] (batch, sequence length, vocab size).
        target_ids: Tensor of shape [N, L] (token IDs).
        
    Returns:
        token_log_probs: Tensor of shape [N, L] containing log pi(x_t | x_<t).
    """
    if logits.dim() != 3 or target_ids.dim() != 2:
        raise ValueError(
            f"Shape mismatch: logits {logits.shape} must be 3D, target_ids {target_ids.shape} must be 2D"
        )
    
    # Compute log probabilities in log-space to prevent underflow
    log_probs = F.log_softmax(logits, dim=-1)  # [N, L, V]
    
    # Vectorized gather: index into vocabulary dimension
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)  # [N, L]
    
    return token_log_probs


def grpo_surrogate_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    clip_eps: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes the clipped surrogate policy gradient loss for GRPO.
    
    Formula:
        r_t = exp(log_pi(y_t) - log_pi_old(y_t))
        L_clip = - mean_i( 1/|o_i| sum_t min(r_t * A_i, clip(r_t, 1-eps, 1+eps) * A_i) )
    
    Args:
        log_probs: [N, L] Log probabilities under current policy pi_theta.
        old_log_probs: [N, L] Log probabilities under sampling policy pi_old (detached).
        advantages: [N] Group-relative advantages for each sequence.
        completion_mask: [N, L] Binary mask (1.0 for valid generated tokens, 0.0 for padding/prompt).
        clip_eps: PPO/GRPO clipping parameter (typically 0.2).
        
    Returns:
        loss: Scalar tensor for backpropagation.
        metrics: Dictionary with diagnostics (clip_fraction, approx_kl, mean_ratio).
    """
    # Expand advantages to [N, 1] for token-level broadcasting
    adv = advantages.unsqueeze(-1)  # [N, 1]

    # Importance sampling ratio r_t
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)  # [N, L]

    # Clipped surrogate objective
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_token_loss = -torch.min(surr1, surr2)  # [N, L]

    # Masked token-level aggregation normalized by sequence length
    masked_loss = per_token_loss * completion_mask
    completion_lengths = completion_mask.sum(dim=-1).clamp(min=1.0)  # [N]
    per_sequence_loss = masked_loss.sum(dim=-1) / completion_lengths  # [N]

    loss = per_sequence_loss.mean()

    # Empirical diagnostics
    with torch.no_grad():
        clipped = (ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))
        clip_fraction = (clipped.float() * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        approx_kl = 0.5 * ((ratio - 1.0) ** 2 * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        mean_ratio = (ratio * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    metrics = {
        "clip_fraction": float(clip_fraction.item()),
        "approx_kl": float(approx_kl.item()),
        "mean_ratio": float(mean_ratio.item()),
    }

    return loss, metrics


def compute_schulman_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes Schulman's k3 unbiased estimator for KL divergence D_KL(pi || pi_ref).
    
    Formula:
        ratio = exp(ref_log_probs - log_probs) = pi_ref / pi
        k3_kl = ratio - log(ratio) - 1
    
    Args:
        log_probs: [N, L] Current policy log probabilities.
        ref_log_probs: [N, L] Reference policy log probabilities (frozen base model).
        completion_mask: [N, L] Mask for active completion tokens.
        
    Returns:
        kl_div: Non-negative scalar KL estimate.
    """
    log_ratio = ref_log_probs - log_probs  # log(pi_ref / pi)
    ratio = torch.exp(log_ratio)           # pi_ref / pi
    k3_kl = ratio - log_ratio - 1.0        # k3 estimator: always >= 0

    masked_kl = (k3_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    return masked_kl

