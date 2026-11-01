"""
Deterministic Rule-Based Reasoning Verifier.
Extracts chain-of-thought traces, validates formatting constraints, and verifies
numerical / symbolic ground-truth answers without learned reward models.
"""

import re
from typing import Dict, List, Tuple


def extract_boxed_content(text: str) -> str:
    """Extracts content inside LaTeX \\boxed{...} with balanced brace matching."""
    idx = text.find(r"\boxed{")
    if idx == -1:
        return ""
    
    start = idx + len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return ""


def extract_tagged_answer(text: str, tag: str = "answer") -> str:
    """Extracts content inside XML-style tags like <answer>...</answer>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return ""


def check_format_compliance(text: str) -> float:
    """
    Rewards structured reasoning: must contain <think>...</think> and an answer block.
    Returns:
        1.0 if full format is respected.
        0.5 if only answer block is present.
        0.0 if completely malformed.
    """
    has_think = bool(re.search(r"<think>.*?</think>", text, flags=re.DOTALL))
    has_answer = bool(extract_tagged_answer(text) or extract_boxed_content(text))

    if has_think and has_answer:
        return 1.0
    elif has_answer:
        return 0.5
    return 0.0


def compute_reasoning_reward(
    completion: str,
    ground_truth: str,
    format_weight: float = 0.2,
    accuracy_weight: float = 0.8,
) -> Tuple[float, Dict[str, float]]:
    """
    Computes total deterministic reward R_i in [0.0, 1.0].
    
    R_i = w_acc * Accuracy + w_fmt * Format
    """
    # 1. Format Score
    fmt_score = check_format_compliance(completion)

    # 2. Extract Candidate Solution
    predicted = extract_tagged_answer(completion)
    if not predicted:
        predicted = extract_boxed_content(completion)

    # Clean whitespace and trailing punctuations
    norm_pred = predicted.strip().strip(".$#").strip()
    norm_target = ground_truth.strip().strip(".$#").strip()

    # Numerical / Exact String Equivalence
    acc_score = 0.0
    if norm_pred and norm_target:
        if norm_pred == norm_target:
            acc_score = 1.0
        else:
            try:
                # Attempt float conversion for numerical tolerance
                if abs(float(norm_pred) - float(norm_target)) < 1e-5:
                    acc_score = 1.0
            except ValueError:
                pass

    total_reward = (accuracy_weight * acc_score) + (format_weight * fmt_score)

    diagnostics = {
        "accuracy_score": acc_score,
        "format_score": fmt_score,
        "total_reward": total_reward,
        "extracted_answer": norm_pred,
    }

    return total_reward, diagnostics

