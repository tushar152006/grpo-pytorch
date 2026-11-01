"""
Hugging Face Pretrained Model Adapter for GRPO.
Enables seamless integration between open-weight pretrained LLMs
(e.g., SmolLM-135M, Qwen2.5-0.5B) and our native PyTorch GRPO engine.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


def load_policy_and_reference_models(
    model_name: str = "HuggingFaceTB/SmolLM-135M",
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float32,
) -> Tuple[nn.Module, nn.Module, PreTrainedTokenizerBase]:
    """
    Loads an open-weights causal language model and instantiates:
    1. Trainable Policy Model (pi_theta)
    2. Completely Frozen Reference Model (pi_ref)
    3. Aligned Tokenizer with valid padding configurations
    
    Args:
        model_name: Hugging Face model repository ID.
        device: Device to place models ('cuda', 'cpu', or auto-detected).
        torch_dtype: Precision format (e.g. torch.bfloat16, torch.float32).
        
    Returns:
        policy_model: Trainable nn.Module
        ref_model: Frozen nn.Module (requires_grad = False)
        tokenizer: PreTrainedTokenizerBase
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[HF_ADAPTER] Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Standard Decoder-Only LLM Padding Convention:
    # Pretrained causal models lack an explicit pad token; map pad_token to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[HF_ADAPTER] Loading policy model ({model_name}) in {torch_dtype} on {device}...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to(device)

    print(f"[HF_ADAPTER] Loading frozen reference model ({model_name})...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to(device)

    # Freeze reference model completely
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return policy_model, ref_model, tokenizer


def format_reasoning_prompt(question: str) -> str:
    """
    Formats a user question into a standard Chain-of-Thought reasoning prompt.
    Elicits structured thinking within <think> tags and boxed answers.
    """
    return (
        f"Problem: {question}\n"
        "Solve this problem carefully. First, write your detailed thought process "
        "inside <think>...</think> tags. Finally, output your exact answer inside \\boxed{...}.\n"
        "Solution: <think>"
    )


def batch_tokenize_prompts(
    prompts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes and pads a batch of prompt strings into synchronized tensors.
    
    Returns:
        prompt_ids:  [B, L_prompt]
        prompt_mask: [B, L_prompt]
    """
    encodings = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    prompt_ids = encodings["input_ids"].to(device)
    prompt_mask = encodings["attention_mask"].to(device)
    return prompt_ids, prompt_mask
