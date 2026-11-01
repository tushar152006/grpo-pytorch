"""
GRPO Batched Rollout Sampler Module.
Generates G reasoning paths per prompt, handles dynamic padding, builds
exact completion masks, and records reference log-probabilities.
Strict native PyTorch implementation without high-level library dependencies.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from src.loss import compute_token_log_probs


def repeat_prompts_for_group(
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Duplicates each prompt G times for group-relative sampling.
    
    Dimensions:
        Input:
            prompt_ids:  [B, L_prompt]
            prompt_mask: [B, L_prompt]
        Output:
            repeated_ids:  [B * G, L_prompt]
            repeated_mask: [B * G, L_prompt]
            
    Ordering:
        Interleaved so that completions for prompt b are contiguous:
        [p0_g0, p0_g1, ..., p0_gG-1, p1_g0, p1_g1, ..., p1_gG-1]
    """
    B, L = prompt_ids.shape
    # Repeat interleave ensures group elements stay adjacent: [B, G, L] -> [B * G, L]
    repeated_ids = prompt_ids.repeat_interleave(group_size, dim=0)
    repeated_mask = prompt_mask.repeat_interleave(group_size, dim=0)
    return repeated_ids, repeated_mask


def generate_group_rollouts(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    group_size: int,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_p: float = 0.95,
    eos_token_id: Optional[int] = None,
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Autoregressively generates G completions per prompt in a single batched tensor pass.
    
    Dimensions:
        prompt_ids: [B, L_prompt]
        Returns:
            all_ids:         [B * G, L_prompt + L_gen] (full prompt + completion sequence)
            completion_mask: [B * G, L_prompt + L_gen] (1.0 for valid generated tokens, 0.0 otherwise)
            old_log_probs:   [B * G, L_prompt + L_gen] (log probabilities under sampling policy)
    """
    model.eval()
    device = prompt_ids.device
    B, L_prompt = prompt_ids.shape
    N = B * group_size  # Total parallel rollouts

    # 1. Expand prompts for group size G: [B * G, L_prompt]
    curr_ids, curr_mask = repeat_prompts_for_group(prompt_ids, prompt_mask, group_size)

    # Track which sequences in the batch have hit EOS: [N]
    is_finished = torch.zeros(N, dtype=torch.bool, device=device)

    # 2. Autoregressive Rollout Loop (strictly under torch.no_grad to save VRAM)
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass through model to obtain next-token logits: [N, L_current, V]
            outputs = model(curr_ids, attention_mask=curr_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            next_token_logits = logits[:, -1, :]  # [N, V]

            # Temperature scaling
            if temperature > 0.0:
                scaled_logits = next_token_logits / temperature
            else:
                scaled_logits = next_token_logits

            # Nucleus (top-p) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above top_p (shift right by 1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                # Scatter back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                scaled_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample next token
            probs = F.softmax(scaled_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [N]

            # If a sequence already finished, replace with pad_token_id
            if eos_token_id is not None:
                next_tokens = torch.where(is_finished, torch.full_like(next_tokens, pad_token_id), next_tokens)
                is_finished = is_finished | (next_tokens == eos_token_id)

            # Append to sequence
            curr_ids = torch.cat([curr_ids, next_tokens.unsqueeze(-1)], dim=-1)
            next_mask = (~is_finished).long().unsqueeze(-1) if eos_token_id is not None else torch.ones((N, 1), dtype=torch.long, device=device)
            curr_mask = torch.cat([curr_mask, next_mask], dim=-1)

            # Early stopping if all sequences in batch hit EOS
            if eos_token_id is not None and is_finished.all():
                break

    # 3. Construct exact completion mask: [N, L_total]
    # 0.0 on prompt tokens, 1.0 on generated tokens prior to EOS/padding
    L_total = curr_ids.shape[1]
    completion_mask = torch.zeros((N, L_total), dtype=torch.float32, device=device)
    
    # Active tokens in generation window
    completion_mask[:, L_prompt:] = curr_mask[:, L_prompt:].float()

    # 4. Compute sampling policy log-probabilities pi_old across the entire sequence
    with torch.no_grad():
        outputs = model(curr_ids, attention_mask=curr_mask)
        full_logits = outputs.logits if hasattr(outputs, "logits") else outputs  # [N, L_total, V]
        
        # Shift logits and targets by 1 for next-token prediction
        shift_logits = full_logits[:, :-1, :].contiguous()  # [N, L_total - 1, V]
        shift_targets = curr_ids[:, 1:].contiguous()        # [N, L_total - 1]
        
        step_log_probs = compute_token_log_probs(shift_logits, shift_targets)  # [N, L_total - 1]
        
        # Pad front with zero for alignment with curr_ids: [N, L_total]
        old_log_probs = torch.cat(
            [torch.zeros((N, 1), device=device), step_log_probs],
            dim=-1,
        )

    return {
        "all_ids": curr_ids,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,
        "group_size": torch.tensor(group_size),
        "prompt_length": torch.tensor(L_prompt),
    }

