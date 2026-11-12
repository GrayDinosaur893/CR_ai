"""
Sample script to fine-tune a small LLM (e.g. Qwen/Llama) using HuggingFace TRL & PEFT (LoRA).
Can be run on Google Colab, Kaggle, or local GPU.
"""

import os

def check_requirements():
    print("=== Model Fine-Tuning Setup ===")
    print("Dependencies required for training:")
    print("  pip install torch transformers datasets trl peft accelerate bitsandbytes")
    print("\nData sources:")
    for train_file in [
        os.path.join("data", "derived", "ryley_finetune_train.jsonl"),
        os.path.join("data", "derived", "finetune_train_sample.jsonl"),
    ]:
        if os.path.exists(train_file):
            with open(train_file, "r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
            print(f"  [Found] {train_file} ({count} SFT examples)")
        else:
            print(f"  [Missing] {train_file}")

    policy_ckpt = os.path.join("models", "ryley_elixir_policy.pt")
    if os.path.exists(policy_ckpt):
        print(f"  [Found] Trained PyTorch policy checkpoint: {policy_ckpt}")

    print("\nTo fine-tune full LLM LoRA (when CUDA + peft/trl/datasets/bitsandbytes are installed):")
    print("  1. Rebuild Ryley dataset: python scripts/prepare_ryley_training.py")
    print("  2. Train local CPU/GPU policy net: python scripts/train_ryley_policy.py")
    print("  3. Run SFTTrainer on GPU using data/derived/ryley_finetune_train.jsonl (e.g. 'Qwen/Qwen2.5-7B-Instruct').")

if __name__ == "__main__":
    check_requirements()
