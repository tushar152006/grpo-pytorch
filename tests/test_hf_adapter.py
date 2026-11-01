"""
Unit tests for Hugging Face pretrained adapter.
Validates prompt formatting, tokenization padding, and attention mask alignment.
"""

import pytest
import torch
from unittest.mock import MagicMock
from src.hf_adapter import format_reasoning_prompt, batch_tokenize_prompts


def test_format_reasoning_prompt():
    question = "What is 15 * 6?"
    prompt = format_reasoning_prompt(question)

    assert "What is 15 * 6?" in prompt
    assert "<think>" in prompt
    assert r"\boxed{...}" in prompt
    assert prompt.endswith("<think>")


def test_batch_tokenize_prompts_mock():
    # Mock tokenizer to verify tensor alignment without requiring network download during unit tests
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[10, 20, 0], [10, 20, 30]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
    }

    prompts = ["Short prompt", "A longer prompt here"]
    device = torch.device("cpu")

    prompt_ids, prompt_mask = batch_tokenize_prompts(prompts, mock_tokenizer, device)

    assert prompt_ids.shape == (2, 3)
    assert prompt_mask.shape == (2, 3)
    # Mask must be 0 for padding token
    assert prompt_mask[0, 2] == 0
    assert prompt_mask[1, 2] == 1
