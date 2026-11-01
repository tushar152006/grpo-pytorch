"""
Unit tests for deterministic rule-based reasoning verifier.
"""

from src.verifier import (
    extract_boxed_content,
    extract_tagged_answer,
    check_format_compliance,
    compute_reasoning_reward,
)


def test_extract_boxed_content_nested_braces():
    text = r"Therefore, the solution simplifies to \boxed{\frac{14}{5}} after calculation."
    assert extract_boxed_content(text) == r"\frac{14}{5}"


def test_extract_tagged_answer():
    text = "<think>Let me compute 25 * 4 = 100.</think><answer>100</answer>"
    assert extract_tagged_answer(text) == "100"


def test_check_format_compliance():
    full_format = "<think>Reasoning step</think><answer>42</answer>"
    assert check_format_compliance(full_format) == 1.0

    missing_think = "The answer is <answer>42</answer>"
    assert check_format_compliance(missing_think) == 0.5

    unstructured = "The answer is 42."
    assert check_format_compliance(unstructured) == 0.0


def test_compute_reasoning_reward_numerical_equivalence():
    completion = "<think>Calculating square root of 49.</think><answer>7.0</answer>"
    reward, diag = compute_reasoning_reward(completion, ground_truth="7")
    
    # Accuracy = 1.0 (float 7.0 == 7), Format = 1.0
    # Total = 0.8 * 1.0 + 0.2 * 1.0 = 1.0
    assert reward == 1.0
    assert diag["accuracy_score"] == 1.0
    assert diag["format_score"] == 1.0


def test_compute_reasoning_reward_wrong_answer():
    completion = "<think>Calculation error.</think><answer>99</answer>"
    reward, diag = compute_reasoning_reward(completion, ground_truth="42")
    
    # Accuracy = 0.0, Format = 1.0 -> Total = 0.2
    assert reward == 0.2
    assert diag["accuracy_score"] == 0.0
    assert diag["format_score"] == 1.0

