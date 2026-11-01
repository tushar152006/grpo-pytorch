"""
GRPO Online Policy Trainer Module.
Orchestrates rollout generation, deterministic reward computation, group advantage
estimation, clipped surrogate optimization, and KL regularization.
Strict native PyTorch implementation without high-level library dependencies.
"""

from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from src.loss import (
    compute_group_advantages,
    compute_schulman_kl,
    compute_token_log_probs,
    grpo_surrogate_loss,
)
from src.sampler import generate_group_rollouts
from src.verifier import compute_reasoning_reward


class GRPOTrainer:
    """
    Online Reinforcement Learning Trainer implementing Group Relative Policy Optimization.
    """
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: Optional[nn.Module],
        optimizer: torch.optim.Optimizer,
        group_size: int = 4,
        max_new_tokens: int = 32,
        clip_eps: float = 0.2,
        kl_weight: float = 0.04,
        max_grad_norm: float = 1.0,
        temperature: float = 0.8,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        pad_token_id: int = 0,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.top_p = top_p
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # Freeze reference model completely to save computation and prevent accidental drift
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        ground_truths: List[str],
        decode_fn: Callable[[List[int]], str],
    ) -> Dict[str, float]:
        """
        Executes a single end-to-end online GRPO training iteration.

        Args:
            prompt_ids: Tensor of shape [B, L_prompt]
            prompt_mask: Tensor of shape [B, L_prompt]
            ground_truths: List of B ground-truth answers for reward verification
            decode_fn: Callable mapping token ID list to decoded text string

        Returns:
            metrics: Dictionary of training telemetry and diagnostic scalars
        """
        device = prompt_ids.device
        B, L_prompt = prompt_ids.shape
        N = B * self.group_size

        # -----------------------------------------------------------------
        # Phase 1: Environment Rollout Generation (under torch.no_grad)
        # -----------------------------------------------------------------
        rollout = generate_group_rollouts(
            model=self.policy_model,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            group_size=self.group_size,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

        all_ids = rollout["all_ids"]                  # [N, L_total]
        completion_mask = rollout["completion_mask"]  # [N, L_total]
        old_log_probs = rollout["old_log_probs"]      # [N, L_total]

        # -----------------------------------------------------------------
        # Phase 2: Deterministic Rule-Based Reward Computation
        # -----------------------------------------------------------------
        rewards_list: List[float] = []
        format_scores: List[float] = []
        accuracy_scores: List[float] = []

        for i in range(N):
            prompt_idx = i // self.group_size
            gt = ground_truths[prompt_idx]

            # Extract generated token IDs
            generated_ids = all_ids[i, L_prompt:].tolist()
            text = decode_fn(generated_ids)

            reward, diag = compute_reasoning_reward(text, ground_truth=gt)
            rewards_list.append(reward)
            format_scores.append(diag["format_score"])
            accuracy_scores.append(diag["accuracy_score"])

        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=device)  # [N]

        # -----------------------------------------------------------------
        # Phase 3: Group-Relative Advantage Normalization
        # -----------------------------------------------------------------
        advantages = compute_group_advantages(
            rewards=rewards_tensor,
            group_size=self.group_size,
        )  # [N]

        # -----------------------------------------------------------------
        # Phase 4: Policy Forward Pass & Gradient Computation
        # -----------------------------------------------------------------
        self.policy_model.train()
        
        # Build attention mask for entire sequence
        full_attention_mask = (all_ids != self.pad_token_id).long()
        outputs = self.policy_model(all_ids, attention_mask=full_attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs  # [N, L_total, V]

        # Shift by 1 for causal autoregressive prediction
        shift_logits = logits[:, :-1, :].contiguous()  # [N, L_total - 1, V]
        shift_targets = all_ids[:, 1:].contiguous()    # [N, L_total - 1]

        step_log_probs = compute_token_log_probs(shift_logits, shift_targets)  # [N, L_total - 1]
        
        # Prepend zero to maintain alignment with completion_mask: [N, L_total]
        curr_log_probs = torch.cat(
            [torch.zeros((N, 1), device=device), step_log_probs],
            dim=-1,
        )

        # -----------------------------------------------------------------
        # Phase 5: Loss Aggregation (Surrogate + Schulman KL)
        # -----------------------------------------------------------------
        surr_loss, loss_metrics = grpo_surrogate_loss(
            log_probs=curr_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            completion_mask=completion_mask,
            clip_eps=self.clip_eps,
        )

        # Optional KL penalty against reference policy
        if self.ref_model is not None and self.kl_weight > 0.0:
            with torch.no_grad():
                ref_outputs = self.ref_model(all_ids, attention_mask=full_attention_mask)
                ref_logits = ref_outputs.logits if hasattr(ref_outputs, "logits") else ref_outputs
                shift_ref_logits = ref_logits[:, :-1, :].contiguous()
                step_ref_log_probs = compute_token_log_probs(shift_ref_logits, shift_targets)
                ref_log_probs = torch.cat(
                    [torch.zeros((N, 1), device=device), step_ref_log_probs],
                    dim=-1,
                )

            kl_penalty = compute_schulman_kl(
                log_probs=curr_log_probs,
                ref_log_probs=ref_log_probs,
                completion_mask=completion_mask,
            )
            total_loss = surr_loss + self.kl_weight * kl_penalty
        else:
            kl_penalty = torch.tensor(0.0, device=device)
            total_loss = surr_loss

        # -----------------------------------------------------------------
        # Phase 6: Backpropagation & Optimization
        # -----------------------------------------------------------------
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping to preserve numerical stability
        grad_norm = nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            max_norm=self.max_grad_norm,
        )

        self.optimizer.step()

        # Telemetry
        metrics = {
            "loss/total": float(total_loss.item()),
            "loss/surrogate": float(surr_loss.item()),
            "loss/kl": float(kl_penalty.item()),
            "reward/mean": float(rewards_tensor.mean().item()),
            "reward/max": float(rewards_tensor.max().item()),
            "accuracy/mean": float(sum(accuracy_scores) / len(accuracy_scores)),
            "format/mean": float(sum(format_scores) / len(format_scores)),
            "policy/clip_fraction": loss_metrics["clip_fraction"],
            "policy/grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
        }

        return metrics

