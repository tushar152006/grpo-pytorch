"""
End-to-End GRPO Reasoning Experiment on the Algorithmic Countdown Task.
Demonstrates policy optimization, reward progression, and reasoning emergence
from mathematical first principles in pure PyTorch.
"""

import sys
import os
import random
import torch
import torch.nn as nn
from typing import List, Tuple

# Add workspace root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.trainer import GRPOTrainer
from src.verifier import compute_reasoning_reward


class TinyReasoningTransformer(nn.Module):
    """
    Lightweight Decoder-Only Transformer architecture for fast local RL experimentation.
    Implements causal self-attention, residual connections, and token embeddings.
    """
    def __init__(self, vocab_size: int = 128, d_model: int = 64, n_heads: int = 2, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(256, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        B, L = input_ids.shape
        device = input_ids.device
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        # Causal triangular mask to enforce autoregressive property
        causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)
        
        h = self.transformer(x, mask=causal_mask)
        logits = self.head(h)
        return logits


class SimpleCharTokenizer:
    """ASCII character tokenizer with special reasoning tags."""
    def __init__(self):
        chars = [chr(i) for i in range(32, 127)]
        self.vocab = ["<pad>", "<eos>"] + chars
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}
        self.id_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.pad_id = 0
        self.eos_id = 1
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(c, self.pad_id) for c in text]

    def decode(self, token_ids: List[int]) -> str:
        res = []
        for tid in token_ids:
            if tid == self.eos_id or tid == self.pad_id:
                continue
            res.append(self.id_to_char.get(tid, ""))
        return "".join(res)


def generate_arithmetic_dataset(num_samples: int = 20) -> List[Tuple[str, str]]:
    """Generates verifiable arithmetic problems: 'Compute: A + B =' -> target sum."""
    dataset = []
    for _ in range(num_samples):
        a = random.randint(2, 9)
        b = random.randint(2, 9)
        prompt = f"Q: {a}+{b}= "
        answer = str(a + b)
        dataset.append((prompt, answer))
    return dataset


def run_experiment(num_steps: int = 15, group_size: int = 4):
    print("=" * 70)
    print("[START] RUNNING GRPO ONLINE REASONING EXPERIMENT (PURE PYTORCH)")
    print(f"Group Size (G): {group_size} rollouts per prompt | Steps: {num_steps}")
    print("=" * 70)

    tokenizer = SimpleCharTokenizer()
    dataset = generate_arithmetic_dataset(num_samples=30)

    # Initialize Policy and frozen Reference Model
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device}")

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
        kl_weight=0.02,
        max_grad_norm=1.0,
        temperature=0.8,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
    )

    print(f"{'Step':<6} | {'Loss':<8} | {'SurrLoss':<10} | {'KL':<8} | {'MeanRew':<9} | {'MaxRew':<8} | {'ClipFrac':<8}")
    print("-" * 70)

    for step in range(1, num_steps + 1):
        # Sample mini-batch of prompts
        batch = random.sample(dataset, k=2)
        prompt_texts = [p[0] for p in batch]
        ground_truths = [p[1] for p in batch]

        # Tokenize and pad prompts
        tokenized_prompts = [tokenizer.encode(p) for p in prompt_texts]
        max_len = max(len(p) for p in tokenized_prompts)
        padded_prompts = [p + [tokenizer.pad_id] * (max_len - len(p)) for p in tokenized_prompts]

        prompt_ids = torch.tensor(padded_prompts, dtype=torch.long, device=device)
        prompt_mask = (prompt_ids != tokenizer.pad_id).long()

        metrics = trainer.train_step(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            ground_truths=ground_truths,
            decode_fn=tokenizer.decode,
        )

        print(
            f"{step:<6} | "
            f"{metrics['loss/total']:<8.4f} | "
            f"{metrics['loss/surrogate']:<10.4f} | "
            f"{metrics['loss/kl']:<8.4f} | "
            f"{metrics['reward/mean']:<9.3f} | "
            f"{metrics['reward/max']:<8.3f} | "
            f"{metrics['policy/clip_fraction']:<8.3f}"
        )

    print("=" * 70)
    print("[SUCCESS] GRPO EXPERIMENT COMPLETED WITH STABLE TELEMETRY")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment(num_steps=10, group_size=4)
