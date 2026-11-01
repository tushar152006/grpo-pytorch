"""
Systematic Ablation Study Engine for GRPO.
Evaluates:
1. Impact of Group Sampling Budget (G in {2, 4, 8}) on reward variance and stability.
2. Impact of Schulman KL Regularization (beta in {0.0, 0.04, 0.20}) on policy drift.
Generates publication-ready figures saved in assets/.
"""

import sys
import os
import random
from typing import Dict, List
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Headless backend for server/script execution
import matplotlib.pyplot as plt

# Add workspace root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_countdown import (
    TinyReasoningTransformer,
    SimpleCharTokenizer,
    generate_arithmetic_dataset,
)
from src.trainer import GRPOTrainer


def run_single_sweep(
    group_size: int,
    kl_weight: float,
    num_steps: int = 10,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Runs a single controlled training trajectory and returns time-series telemetry."""
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleCharTokenizer()
    dataset = generate_arithmetic_dataset(num_samples=40)

    policy_model = TinyReasoningTransformer(vocab_size=tokenizer.vocab_size).to(device)
    ref_model = TinyReasoningTransformer(vocab_size=tokenizer.vocab_size).to(device)
    ref_model.load_state_dict(policy_model.state_dict())

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-3, weight_decay=0.01)

    trainer = GRPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        optimizer=optimizer,
        group_size=group_size,
        max_new_tokens=24,
        clip_eps=0.2,
        kl_weight=kl_weight,
        max_grad_norm=1.0,
        temperature=0.8,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
    )

    history = {
        "step": [],
        "loss": [],
        "kl": [],
        "reward_mean": [],
        "clip_fraction": [],
    }

    for step in range(1, num_steps + 1):
        batch = random.sample(dataset, k=2)
        prompt_texts = [p[0] for p in batch]
        ground_truths = [p[1] for p in batch]

        tokenized = [tokenizer.encode(p) for p in prompt_texts]
        max_len = max(len(p) for p in tokenized)
        padded = [p + [tokenizer.pad_id] * (max_len - len(p)) for p in tokenized]

        prompt_ids = torch.tensor(padded, dtype=torch.long, device=device)
        prompt_mask = (prompt_ids != tokenizer.pad_id).long()

        metrics = trainer.train_step(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            ground_truths=ground_truths,
            decode_fn=tokenizer.decode,
        )

        history["step"].append(step)
        history["loss"].append(metrics["loss/total"])
        history["kl"].append(metrics["loss/kl"])
        history["reward_mean"].append(metrics["reward/mean"])
        history["clip_fraction"].append(metrics["policy/clip_fraction"])

    return history


def generate_ablation_plots(output_dir: str = "assets"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("[ABLATION] STARTING SYSTEMATIC EMPIRICAL SWEEPS")
    print("=" * 70)

    # -------------------------------------------------------------
    # Sweep 1: Group Size G in {2, 4, 8} with fixed beta = 0.04
    # -------------------------------------------------------------
    print("\n--- Running Sweep 1: Group Size G in {2, 4, 8} ---")
    group_sizes = [2, 4, 8]
    g_results = {}
    for g in group_sizes:
        print(f"  Training with G={g} rollouts/prompt...")
        g_results[g] = run_single_sweep(group_size=g, kl_weight=0.04, num_steps=10)

    # Plot Sweep 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    colors = {2: "#e74c3c", 4: "#2ecc71", 8: "#3498db"}

    for g in group_sizes:
        steps = g_results[g]["step"]
        ax1.plot(steps, g_results[g]["reward_mean"], marker="o", color=colors[g], label=f"G = {g} rollouts")
        ax2.plot(steps, g_results[g]["clip_fraction"], marker="s", color=colors[g], label=f"G = {g} rollouts")

    ax1.set_title("Mean Reward vs. Group Budget (G)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Step", fontsize=10)
    ax1.set_ylabel("Mean Reward", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2.set_title("Token Clipping Fraction vs. Group Budget (G)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Training Step", fontsize=10)
    ax2.set_ylabel("Clip Fraction", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "ablation_group_size.png")
    fig.savefig(plot1_path)
    plt.close(fig)
    print(f"  [SAVED] {plot1_path}")

    # -------------------------------------------------------------
    # Sweep 2: KL Regularization beta in {0.0, 0.04, 0.20} with fixed G = 4
    # -------------------------------------------------------------
    print("\n--- Running Sweep 2: KL Regularization beta in {0.0, 0.04, 0.20} ---")
    betas = [0.0, 0.04, 0.20]
    beta_results = {}
    for b in betas:
        print(f"  Training with beta={b}...")
        beta_results[b] = run_single_sweep(group_size=4, kl_weight=b, num_steps=10)

    # Plot Sweep 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    beta_colors = {0.0: "#e67e22", 0.04: "#27ae60", 0.20: "#8e44ad"}

    for b in betas:
        steps = beta_results[b]["step"]
        ax1.plot(steps, beta_results[b]["kl"], marker="^", color=beta_colors[b], label=f"beta = {b}")
        ax2.plot(steps, beta_results[b]["loss"], marker="d", color=beta_colors[b], label=f"beta = {b}")

    ax1.set_title("Schulman KL Divergence Drift vs. Beta", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Step", fontsize=10)
    ax1.set_ylabel("D_KL(pi || pi_ref)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2.set_title("Total Policy Loss vs. Beta", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Training Step", fontsize=10)
    ax2.set_ylabel("Total Objective Loss", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "ablation_kl_drift.png")
    fig.savefig(plot2_path)
    plt.close(fig)
    print(f"  [SAVED] {plot2_path}")

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL ABLATIONS COMPLETED AND HIGH-RES PLOTS GENERATED")
    print("=" * 70)


if __name__ == "__main__":
    generate_ablation_plots(output_dir="assets")
